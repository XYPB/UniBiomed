import json
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask
import numpy as np
import copy

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import torchvision.transforms as T
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from torchvision.transforms.functional import InterpolationMode
from projects.unibiomed.datasets.encode_fn import video_lisa_encode_fn
from projects.unibiomed.datasets.utils import dynamic_preprocess
import random

VQA_QUESTIONS = [
    DEFAULT_IMAGE_TOKEN + 'Can we observe any signs of abnormality in {anatomy}? Please respond with interleaved segmentation masks for the corresponding parts.',
    DEFAULT_IMAGE_TOKEN + 'Can we see abnormality in {anatomy}? Please highlight the region with interleaved segmentation masks.',
    DEFAULT_IMAGE_TOKEN + 'Can abnormality be identified in {anatomy}? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Can abnormality be observed within {anatomy}? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Are there any visible indications of abnormality in {anatomy}? Please respond with interleaved segmentation masks for the corresponding parts.',
    DEFAULT_IMAGE_TOKEN + 'Are there any observable signs of abnormality in {anatomy}? Please highlight the region with interleaved segmentation masks.',
    DEFAULT_IMAGE_TOKEN + 'Does the visual features suggest the presence of abnormality in {anatomy}? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Are there any clear indications of abnormality within {anatomy}? Please output with interleaved segmentation masks for the corresponding phrases.',

]


def parse_annotations(example):
    # example dict_keys(['abnormality', 'image_file', 'mask_file', 'report', 'anatomy'])

    annotations = {'labels': [], 'caption': [], 'abnormality': example['abnormality'], 'masks': example['mask_file'],
                   'anatomy': example['anatomy'],
                   'image': example['image_file'], 'mask_file': example['mask_file']}

    caption = example['abnormality'] + '.'
    annotations['caption'] = caption.lower()

    conversations = process_conversation(annotations['caption'], annotations['anatomy'])
    annotations['conversations'] = conversations

    return annotations


def process_conversation(caption, anatomy):
    # insert <p> </p> and [seg] to caption and select a question
    question = random.choice(VQA_QUESTIONS).strip().format(anatomy=anatomy)

    # Prepare caption with tags
    def tag_caption(caption):
        caption = f"<p> {caption} </p> [SEG]"
        return caption

    detailed_answer = tag_caption(caption)

    conversations = [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': detailed_answer}]
    return conversations


def glamm_map_fn(example):
    # example {'id': str, 'refs': [{"setence", 'bbox', 'segmentation'},], 'img_file_name': str, 'caption': str}

    example = parse_annotations(example)
    # example dict_keys(['abnormality', 'image_file', 'mask_file', 'report', 'anatomy'])

    # do llava preprocess
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    example.update({'conversation': conversation})
    return example


class BiomedVQADataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    def __init__(self,
                 image_folder,
                 json_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
    ):
        super().__init__()
        assert lazy
        self.lazy = lazy
        self.max_length = max_length

        json_data = self.json_file_preprocess(json_path)
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self.image_folder = image_folder

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self.repeats = repeats

        self._system = ''

        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.single_image_mode = single_image_mode

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            if self.lazy:
                cur_len = 100
            else:
                cur_len = len(data_dict['input_ids'])
                if data_dict.get('image', None) is None:
                    cur_len = -cur_len
            length_list.append(cur_len)
        return length_list * self.repeats

    def __len__(self):
        return len(self.text_data) * self.repeats

    def real_len(self):
        return len(self.text_data)

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in object_mask:
                rles = mask.frPyObjects([seg], ori_height, ori_width)
                m = mask.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_map_fn(data_dict)
        return data_dict

    def replace_image_str(self, data_dict, image_str):
        data_dict['conversation'][0]['input'] = \
            data_dict['conversation'][0]['input'].replace(DEFAULT_IMAGE_TOKEN, image_str)
        return data_dict

    def __getitem__(self, index):

        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])

        # parse datasets
        result = self.dataset_map_fn(data_dict)
        data_dict.update(result)

        # process image
        image_file = data_dict['image']
        image = Image.open(os.path.join(self.image_folder,
                                        image_file)).convert('RGB')
        ori_width, ori_height = image.size
        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            data_dict['g_pixel_values'] = g_pixel_values

        if self.single_image_mode:
            images = [image]
        else:
            images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)
        pixel_values = [self.transformer(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        data_dict['pixel_values'] = pixel_values

        num_image_tokens = pixel_values.shape[0] * self.patch_token
        image_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'

        data_dict = self.replace_image_str(data_dict, image_token_str)

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length,
                                      with_image_token=True)
        data_dict.update(result)

        # process mask
        # data_dict['masks'] = self.decode_mask(data_dict['masks'], ori_height=ori_height, ori_width=ori_width)
        masks = np.asarray(Image.open(os.path.join(self.image_folder, data_dict["masks"])))
        if len(masks.shape) != 2:
            masks = masks[:, :, 0]

        if 'no findings' in data_dict['abnormality']:
            h, w = masks.shape
            # if there is no abnormality, masks as zeros
            masks = np.zeros([h, w])

        # Note that m should be 0 and 1
        masks = (masks > 0).astype(np.uint8)
        masks = torch.from_numpy(masks)
        masks = masks.unsqueeze(0)  # shape as (1, h, w) one mask for one caption
        data_dict['masks'] = masks

        if data_dict['masks'] is None:
            return self.__getitem__(0)

        return data_dict


    def json_file_preprocess(self, json_path):
        json_data = json.load(open(json_path))

        output_data = []

        for idx in json_data.keys():
            output_data.append(json_data[idx])
        return output_data