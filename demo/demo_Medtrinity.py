import argparse
import math
import os
import torch
import tqdm
from pycocotools import mask as _mask
from pycocotools.coco import COCO
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
import numpy as np
from demo.eval_utils.utils import _init_dist_pytorch, get_dist_info, collect_results_cpu, get_rank
from PIL import Image
import re
import json
import cv2
import random


Medtrinity_QUESTIONS = [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give about the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]


def parse_args():
    parser = argparse.ArgumentParser(description='UniBiomed Medtrinity')
    parser.add_argument('--model_path', help='hf model path.')
    parser.add_argument(
        '--split',
        default='test',
        help='Specify a split')

    parser.add_argument(
        '--data_path',
        default='./data/Biomed/MedTrinity',
        help='save path')
    parser.add_argument(
        '--annotation_file',
        default='test.json',
        help='save path')

    parser.add_argument(
        '--save_dir',
        default='./val_results/MedTrinity',
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


class MedtrinityInferenceDataset:
    def __init__(self,
                 image_folder,
                 annotation_file=None,
                 ):
        self.image_folder = image_folder
        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())
        import random
        random.shuffle(self.image_dict_keys)

    def __len__(self):
        return len(self.image_dict_keys[:1000])

    def get_questions(self):
        question = random.choice(Medtrinity_QUESTIONS).strip()
        question = "<image>\n" + question
        return question

    def __getitem__(self, index):

        data_dict = {}

        image_id = self.image_dict_keys[index]
        image_file = self.ann_dict[image_id]['file_name']
        questions = self.get_questions()

        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        data_dict['image'] = image
        data_dict['text'] = questions
        data_dict['img_id'] = image_id
        data_dict['answer'] = self.ann_dict[image_id]['caption']

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

    dataset = MedtrinityInferenceDataset(
        image_folder=args.data_path,
        annotation_file=os.path.join(args.data_path, args.annotation_file),
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))

    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        result = {'image_id': data_batch['img_id'], 'image_file': data_batch['image_file'], 'answer': data_batch['answer']}

        del data_batch['img_id'], data_batch['image_file'], data_batch['answer']

        prediction = model.predict_forward(**data_batch, tokenizer=tokenizer)['prediction']

        process_and_save_output(
            args.save_dir,
            result['image_file'],
            prediction,
        )
        results.append(prediction)

    collect_results_cpu(results, len(dataset), tmpdir='./medtrinity_eval_tmp')


def process_and_save_output(output_dir, image_name, text_output):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # Create results dictionary
    # print(f"clean_str: {cleaned_str}")
    result_dict = {
        "image_id": image_name.split('/')[-1][:-4],
        "caption": cleaned_str,
        "phrases": phrases,
    }

    output_path = output_dir + '/' + image_name.split('/')[-1][:-4] + '.json'

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)

    return


if __name__ == '__main__':
    main()

