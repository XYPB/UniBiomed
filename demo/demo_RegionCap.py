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


Region_QUESTIONS = [
    'What is the <region> in the picture?',
    'Can you tell me what is the <region> in the image?',
    'Could you tell me the <region> in the picture?',
    'Can you identify the <region> in the image?',
    'Can you tell me about the <region> represents in this photo?',
    'What does the <region> in the photo indicate?',
    'What does the <region> refer to in this image?',
    'Could you clarify what the <region> in the photo shows?',
    'What is shown in the <region> in the picture?',
]


def parse_args():
    parser = argparse.ArgumentParser(description='UniBiomed Region Understanding')
    parser.add_argument('--model_path', help='hf model path.')
    parser.add_argument(
        '--split',
        default='test',
        help='Specify a split')

    parser.add_argument(
        '--data_path',
        default='./data/Biomed/CHAOS/test',
        help='save path')
    parser.add_argument(
        '--annotation_file',
        default='./data/Biomed/CHAOS/test.json',
        help='save path')

    parser.add_argument(
        '--save_dir',
        default='./val_results/Region_Understand/CHAOS',
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


def box2image(image, mask):
    box_image = np.copy(np.asarray(image))
    box_mask = np.copy(mask)

    contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contours = list(contours)
        random.shuffle(contours)
        selected_contour = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w*h > 400:
                cv2.rectangle(box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                selected_contour = contour
                break

        if selected_contour is None:
            selected_contour = random.choice(contours)
            x, y, w, h = cv2.boundingRect(selected_contour)
            cv2.rectangle(box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    box_image = Image.fromarray(box_image)
    return box_image


def clean_caption(caption):
    modality = caption.split(' ')[-1]
    target = caption.split(' in ')[0]

    # some dataset answers use right/left kidney, some use kidney
    target = target.replace('right ', '')
    target = target.replace('left ', '')

    output = target + ' in ' + modality
    return output


def check(output, answer):
    output, answer = output.lower(), answer.lower()
    modality = answer.split(' ')[-1]
    target = answer.split(' in ')[0]

    output_modality = output.split(' ')[-1]
    output_target = output.split(' in ')[0]

    if target == output_target and modality in output_modality:
        print('true')
        return True
    else:
        print('false')
        return False


class RegionInferenceDataset:
    def __init__(self,
                 image_folder,
                 annotation_file=None,
                 ):
        self.image_folder = image_folder
        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())

    def __len__(self):
        return len(self.image_dict_keys)

    def get_questions(self):
        question = random.choice(Region_QUESTIONS).strip()
        question = "<image>\n" + question
        return question

    def __getitem__(self, index):

        data_dict = {}

        image_id = self.image_dict_keys[index]
        image_file = self.ann_dict[image_id]['file_name']
        mask_file = os.path.join(self.image_folder+'_mask', self.ann_dict[image_id]['mask_file'])
        questions = self.get_questions()

        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        mask = np.asarray(Image.open(mask_file))
        if len(mask.shape) != 2:
            mask = mask[:, :, 0]
        # Note that m should be 0 and 1
        mask = (mask > 0).astype(np.uint8)
        image = box2image(image, mask)

        answer = self.ann_dict[image_id]['sentences'][0]["raw"].lower()
        answer = clean_caption(answer)

        data_dict['image'] = image
        data_dict['text'] = questions
        data_dict['img_id'] = image_id
        data_dict['answer'] = answer

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

    dataset = RegionInferenceDataset(
        image_folder=args.data_path,
        annotation_file=args.annotation_file,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))

    total_number = 0
    correct_number = 0

    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        result_dict = {'image_id': data_batch['img_id'], 'image_file': data_batch['image_file'], 'answer': data_batch['answer']}
        del data_batch['img_id'], data_batch['image_file'], data_batch['answer']

        question = data_batch['text']
        answer = result_dict['answer']

        prediction = model.predict_forward(**data_batch, tokenizer=tokenizer)['prediction']

        text_output = prediction
        text_output = text_output.split("ASSISTANT: ")[-1]
        cleaned_str = re.sub(r'<.*?>', '', text_output)
        cleaned_str = cleaned_str.replace('[SEG]', '')
        cleaned_str = ' '.join(cleaned_str.split()).strip("'")
        cleaned_str = cleaned_str.strip()
        cleaned_str = clean_caption(cleaned_str)

        result_dict["prediction"] = cleaned_str
        results.append(result_dict)

        print(f'User: {question}')
        print(f'Assistant: {cleaned_str}')
        print(f'Answer: {answer}')

        total_number += 1
        correct_flag = check(cleaned_str, answer)
        if correct_flag:
            correct_number += 1

        # save image
        image = data_batch['image']
        save_image_path = os.path.join(args.save_dir, 'box_image')
        os.makedirs(save_image_path, exist_ok=True)
        image.save(os.path.join(save_image_path, result_dict['image_file'][:-4]+'-caption-'+result_dict['answer']+'.png'))

    accuracy = correct_number/total_number
    print('data and accuracy:', args.save_dir.split('/')[-1], accuracy)

    tmpdir = os.path.join(args.save_dir, 'test_temp_regioncap_' + args.model_path.replace('/', '').replace('.', ''))
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    output_path = os.path.join(args.save_dir, 'region_cap_pred_accuracy-'+str(accuracy)+'.json')
    if get_rank() == 0:
        with open(output_path, 'w') as json_file:
            json.dump(results, json_file, indent=2)


if __name__ == '__main__':
    main()

