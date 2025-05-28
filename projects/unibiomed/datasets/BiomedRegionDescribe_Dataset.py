# Copyright (c) OpenMMLab. All rights reserved.
import collections
import os.path
import os.path as osp
import random
from typing import Dict, List

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.dataset import RepeatDataset

import copy
import random

import os
from typing import Literal
import cv2
import torch

from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from xtuner.dataset.utils import encode_fn
from xtuner.dataset.map_fns import llava_map_fn

from projects.glamm.datasets.utils.utils import expand2square
from projects.glamm.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from projects.unibiomed.datasets.utils import dynamic_preprocess
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# these questions for the ResSeg dataset
Region_QUESTIONS = [
    'What is the <region> in the picture? Could you describe the abnormality with details?',
    'Can you tell me what is the <region> in the image? Could you describe the abnormality with details?',
    'Could you tell me the <region> in the picture? Could you describe the abnormality with details?',
    'Can you identify the <region> in the image? Could you describe the abnormality with details?',
    'Can you tell me about the <region> represents in this photo? Could you describe the abnormality with details?',
    'What does the <region> in the photo indicate? Could you describe the abnormality with details?',
    'What does the <region> refer to in this image? Could you describe the abnormality with details?',
    'Could you clarify what the <region> in the photo shows? Could you describe the abnormality with details?',
    'What is shown in the <region> in the picture? Could you describe the abnormality with details?',
]


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


def check_mask_area(mask_name):
    mask = Image.open(mask_name)
    mask = np.asarray(mask)
    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flag = False

    if len(contours) > 0:
        contours = list(contours)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w*h
            if area > 400:
                flag = True
                break

    return flag


class RefBiomedDataset(BaseDataset):
    """RefCOCO dataset.

    The `Refcoco` and `Refcoco+` dataset is based on
    `ReferItGame: Referring to Objects in Photographs of Natural Scenes
    <http://tamaraberg.com/papers/referit.pdf>`_.

    The `Refcocog` dataset is based on
    `Generation and Comprehension of Unambiguous Object Descriptions
    <https://arxiv.org/abs/1511.02283>`_.

    Args:
        ann_file (str): Annotation file path.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str): Prefix for training data.
        split_file (str): Split file path.
        split (str): Split name. Defaults to 'train'.
        text_mode (str): Text mode. Defaults to 'random'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 data_prefix: Dict,
                 text_mode: str = 'random',
                 **kwargs):

        assert text_mode in ['original', 'random', 'concat', 'select_first']
        self.text_mode = text_mode
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        self.instances = mmengine.load(self.ann_file, file_format='json')
        anns, imgs = {}, {}
        for ann in self.instances['annotations']:
            if ann['file_name'] not in anns:
                # judge if this image in anns, if not, initialize
                anns[ann["file_name"]] = []
            anns[ann["file_name"]].append(ann)

        for img in self.instances['images']:
            imgs[img['file_name']] = img

        img_prefix = self.data_prefix['img_path']
        join_path = mmengine.fileio.get_file_backend(img_prefix).join_path

        images_list = [value for key, value in imgs.items()]
        data_list = []

        for image in images_list:
            if image['file_name'] in anns.keys():
                corresponding_anno = anns[image['file_name']]
            else:
                continue

            for ann in corresponding_anno:
                if ann['caption'] is None:
                    continue

                instances = []
                sentences = []

                texts = [x['raw'].lower() for x in ann['sentences']]

                texts = [ann['caption'].lower()]

                # random select one text
                if self.text_mode == 'random':
                    idx = random.randint(0, len(texts) - 1)
                    text = [texts[idx]]
                # concat all texts
                elif self.text_mode == 'concat':
                    text = [''.join(texts)]
                # select the first text
                elif self.text_mode == 'select_first':
                    text = [texts[0]]
                # use all texts
                elif self.text_mode == 'original':
                    text = texts
                else:
                    raise ValueError(f'Invalid text mode "{self.text_mode}".')

                ins = [{
                    'mask': join_path(img_prefix+'_mask', ann['mask_file']),
                    'ignore_flag': 0
                }] * len(text)
                instances.extend(ins)
                sentences.extend(text)

                data_info = {
                    'img_path': join_path(img_prefix, image['file_name']),
                    'img_id': image['file_name'],
                    'instances': instances,
                    'text': sentences
                }
                data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError(f'No sample in split "{self.split}".')

        balance_length = 5000
        if len(data_list) < balance_length:
            repeats = balance_length // len(data_list)
            data_list = data_list * repeats

        return data_list


class RegionRadgenomeBiomedDataset(RefBiomedDataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 data_root,
                 ann_file=None,
                 special_tokens=None,
                 prompt_template=None,
                 data_prefix=dict(img_path='train/'),
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=5,
                 single_image_mode=True,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 repeats=5,
                 **kwargs):

        # repeats: to balance the length of different datasets
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs,
        )
        self.begin_str = f'{DEFAULT_IMAGE_TOKEN}\n'

        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_folder = data_root
        self.template = prompt_template
        self.max_length = max_length

        if self.arch_type == 'intern_vl':
            # self._system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
            self._system = ''
            self.template['INSTRUCTION'] = '<|user|>\n{input}<|end|><|assistant|>\n'
        elif self.arch_type == 'qwen':
            self._system = ''
        elif self.arch_type == 'llava':
            self._system = ''

        self.num_classes_per_sample = num_classes_per_sample
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        if preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)
        self.arch_type = arch_type
        self.single_image_mode = single_image_mode
        self._max_refetch = 1000

        print("Region Understanding dataset, include {} items.".format(len(self)))


    @property
    def modality_length(self):
        ### LengthGroupedSampler !!!
        import pickle
        length_list = []
        for idx in range(len(self)):
            length_list.append(100)
        return length_list

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        masks, phrases = [], []
        instances, text = ann_info['instances'], ann_info['text']

        # index = np.random.choice(range(len(instances)), self.num_classes_per_sample, replace=True)
        index = list(range(len(instances)))
        # print('check index:', len(instances), index)

        for idx in index:
            inst = instances[idx]
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            phrases.append(phrase)
            # print('check mask and annotation:', phrase, inst["mask"])

            binary_mask = np.zeros((height, width), dtype=np.uint8)

            # !!!! Ours are not RLE annotations but mask file, use Image.open to open it
            m = np.asarray(Image.open(inst["mask"]))
            if len(m.shape) != 2:
                m = m[:, :, 0]
            # Note that m should be 0 and 1
            m = (m > 0).astype(np.uint8)
            m = cv2.resize(m, (width, height))
            binary_mask += m
            masks.append(binary_mask)

        conversation = []
        for i, phrase in enumerate(phrases):
            question = random.choice(Region_QUESTIONS).format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            # conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})

            # text as answer !
            conversation.append({'from': 'gpt', 'value': text[0]})

        masks = np.stack([mask for mask in masks], axis=0)[0]

        ann_info.update({
            'masks': masks,
            'conversations': conversation,
            'image': image_path,
            'text': text
        })
        return ann_info

    def prepare_data(self, index):
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {}
        # image file name and conversation
        out_data_dict['image'] = data_dict['image']
        out_data_dict['text'] = data_dict['text']

        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            try:
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                print(f'Error: {e}', flush=True)
                print_log(f'Error: {e}', logger='current')
                return None

            # print(image.size, data_dict['masks'].shape)
            # add box to image !
            # print(data_dict['masks'].shape)
            image = box2image(image, data_dict['masks'])
            out_data_dict['box_image'] = np.asarray(image)

            if self.single_image_mode:
                images = [image]
            else:
                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                            self.max_dynamic_patch,
                                            self.image_size, self.use_thumbnail)
            if self.preprocessor is not None:
                if self.arch_type == 'qwen':
                    _data_dict = self.preprocessor(images, do_resize=True)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                    num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                elif self.arch_type == 'llava':
                    _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    num_image_tokens = _data_dict['pixel_values'].shape[0] * self.patch_token
                else:
                    raise NotImplementedError
                out_data_dict.update(_data_dict)
            else:
                pixel_values = [self.transformer(image) for image in images]
                pixel_values = torch.stack(pixel_values)
                out_data_dict['pixel_values'] = pixel_values

                num_image_tokens = pixel_values.shape[0] * self.patch_token
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            token_dict = self.get_inputid_labels(data_dict['conversations'], image_token_str)
            out_data_dict.update(token_dict)
        else:
            token_dict = self.get_inputid_labels(data_dict['conversations'], None)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(1, 3, self.image_size, self.image_size)

        # Delete masks for language understanding training !
        del out_data_dict['masks']
        return out_data_dict

    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', '')
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')

            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        # print('len_ids: ', len(input_ids))

        return {'input_ids': input_ids, 'labels': labels}

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data


