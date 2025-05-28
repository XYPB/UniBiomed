import argparse
import math
import os
import torch
import tqdm
from pycocotools import mask as mask_utils

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
import numpy as np
from demo.eval_utils.utils import _init_dist_pytorch, get_dist_info, collect_results_cpu
from PIL import Image
import re
import json
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from pycocotools.coco import COCO


QUESTIONS = [
    DEFAULT_IMAGE_TOKEN + 'Can we observe any signs of abnormality? Please respond with interleaved segmentation masks for the corresponding parts.',
    DEFAULT_IMAGE_TOKEN + 'Can we see abnormality? Please highlight the region with interleaved segmentation masks.',
    DEFAULT_IMAGE_TOKEN + 'Can abnormality be identified? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Can abnormality be observed? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Are there any visible indications of abnormality? Please respond with interleaved segmentation masks for the corresponding parts.',
    DEFAULT_IMAGE_TOKEN + 'Are there any observable signs of abnormality? Please highlight the region with interleaved segmentation masks.',
    DEFAULT_IMAGE_TOKEN + 'Does the visual features suggest the presence of abnormality? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Are there any clear indications of abnormality? Please output with interleaved segmentation masks for the corresponding phrases.',

]


def parse_args():
    parser = argparse.ArgumentParser(description='Grounded Report Generation')
    parser.add_argument('--model_path', help='hf model path.')
    parser.add_argument(
        '--split',
        default='test',
        help='Specify a split')

    parser.add_argument(
        '--data_path',
        default='./data/Biomed/LGG',
        help='save path')

    parser.add_argument(
        '--json_path',
        default='test.json',
        help='save path')

    parser.add_argument(
        '--save_dir',
        default='./val_results/Grounded_disease/LGG',
        help='save path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class GroundedLesionInferenceDataset:
    def __init__(self,
                 image_folder, json_file,
                 ):
        self.image_folder = image_folder
        self.coco = COCO(json_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())

    def __len__(self):
        return len(self.image_dict_keys)

    def get_questions(self):
        import random
        question = random.choice(QUESTIONS).strip()
        return question

    def __getitem__(self, index):
        data_dict = {}

        image_id = self.image_dict_keys[index]
        image_file = self.ann_dict[image_id]['file_name']
        question = self.get_questions()

        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')
        data_dict['image'] = image
        data_dict['text'] = question

        data_dict['img_id'] = image_file
        return data_dict


def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    dataset = GroundedLesionInferenceDataset(
        image_folder=os.path.join(args.data_path, args.split),
        json_file=os.path.join(args.data_path, args.json_path),
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        prediction = {'img_id': data_batch['img_id'], 'image_file': data_batch['image_file']}
        del data_batch['img_id'], data_batch['image_file']

        w, h = data_batch['image'].size

        pred_dict = model.predict_forward(**data_batch, tokenizer=tokenizer)
        if 'prediction_masks' not in pred_dict.keys() or pred_dict['prediction_masks'] is None or len(pred_dict['prediction_masks']) == 0:
            print("No SEG !!!")
            prediction['prediction_masks'] = torch.zeros((1, h, w), dtype=torch.bool)

        else:
            masks_torch = [torch.from_numpy(mask) for mask in pred_dict['prediction_masks']]
            prediction['prediction_masks'] = torch.stack(masks_torch, dim=0)[:, 0]

        process_and_save_output(
            args.save_dir,
            prediction['image_file'],
            pred_dict['prediction'],
            prediction['prediction_masks'],
        )
        results.append(pred_dict['prediction'])

    results = collect_results_cpu(results, len(dataset), tmpdir='./grounded_disease_eval_tmp')


def process_and_save_output(output_dir, image_name, text_output, pred_masks):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    # pattern = re.compile(r'<p>(.*?)<\/p>')
    # phrases = pattern.findall(text_output)
    # phrases = [p.strip() for p in phrases]

    # # Remove the [SEG] token
    # cleaned_str = cleaned_str.replace('[SEG]', '')
    #
    # # Strip unnecessary spaces
    # cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    # cleaned_str = cleaned_str.strip()

    cleaned_str = cleaned_str.split('[SEG]. ')[-1]
    phrases = [cleaned_str]
    print(cleaned_str)

    # Convert the predicted masks into RLE format
    pred_masks_tensor = pred_masks.cpu()
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))

    # Create results dictionary
    # print(f"clean_str: {cleaned_str}")
    result_dict = {
        "image_id": image_name[:-4],
        "caption": cleaned_str,
        "phrases": phrases,
        "pred_masks": rle_masks,
    }

    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)

    # save mask for visualization
    mask_numpy = mask_to_numpy(pred_masks_tensor)
    mask = Image.fromarray(mask_numpy)
    output_mask_dir = output_dir + '_mask'
    os.makedirs(output_mask_dir, exist_ok=True)
    mask.save(os.path.join(output_mask_dir, image_name))

    return


def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out


def mask_to_numpy(tensor: torch.Tensor):
    mask_numpy = tensor.data.numpy()[0]
    mask_numpy = (mask_numpy > 0).astype(np.uint8)

    mask_numpy = mask_numpy * 255
    return mask_numpy


def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle


if __name__ == '__main__':
    main()
