import logging
import os
from typing import Literal

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset
from xtuner.dataset.utils import get_bos_eos_token_ids
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
import copy

import json
import random
import pycocotools.mask as maskUtils
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

SEG_QUESTIONS = [
    "Can you segment the {class_name}?",
    "Please segment {class_name}.",
    "What is {class_name}? Please respond with segmentation mask.",
    "What is {class_name}? Please output segmentation mask.",

    "Can you segment the {class_name}.",
    "Please segment {class_name}.",
    "What is {class_name}? Please respond with segmentation mask.",
    "What is {class_name}? Please output segmentation mask.",

    "Could you provide a segmentation mask for the {class_name}?",
    "Please identify and segment the {class_name}.",
    "Where is the {class_name}? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name}?",
    "Please identify and segment the {class_name}.",
    "Where is the {class_name}? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} with a segmentation mask?",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]


def volume_lisa_encode_fn(
        example,
        tokenizer,
        max_length,
        input_ids_with_output=True,
        **kwargs
):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output
            output_with_loss = single_turn_conversation.get(
                'output_with_loss', True)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}


def delete_slices(meta):
    new_meta = meta.copy()
    new_meta_slices = []

    for slice_name in meta['slices']:
        add_flag = False
        for mask_name in meta['mask_file']:
            if slice_name[:-4] == mask_name.split('/')[-1][:len(slice_name[:-4])]:
                # this slice contain mask
                add_flag = True
                break

        if add_flag is True:
            new_meta_slices.append(slice_name)

    new_meta['slices'] = new_meta_slices
    return new_meta


class ReferSegBiomedDataset3D(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    FAST_IMG_CONTEXT_TOKEN = '<FAST_IMG_CONTEXT>'
    FAST_IMG_START_TOKEN = '<fast_img>'
    FAST_IMG_END_TOKEN = '</fast_img>'

    def __init__(self,
                 image_folder,
                 expression_file,
                 extra_image_processor=None,
                 tokenizer=None,
                 select_number=10,  # how many classes to seg in one volume
                 sampled_slices=10,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=2048,
                 lazy=True,
                 repeats=1,
                 special_tokens=None,
                 frame_contiguous_sample=False,
                 use_fast=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 # only work if use_fast = True
                 n_fast_images=50,
                 fast_pool_size=4,
                 fast_token_after_question=False,
    ):
        assert lazy is True
        self.tokenizer = BUILDER.build(tokenizer)
        self.select_number = select_number
        self.sampled_slices = sampled_slices
        assert offline_processed_text_folder or (expression_file and tokenizer)
        self.lazy = lazy

        self.max_length = max_length

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if offline_processed_text_folder and expression_file:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            vid2metaid, metas = self.json_file_preprocess(expression_file)
            self.vid2metaid = vid2metaid

            self.volumes = list(self.vid2metaid.keys())
            self.json_datas = metas
            json_datas = metas
            json_data = DatasetDict({'train': HFDataset.from_list(json_datas)})
            if self.lazy:
                self.text_data = build_origin_dataset(json_data, 'train')
            else:
                raise NotImplementedError

        self.image_folder = image_folder
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.down_ratio = 1
        self.repeats = repeats

        self._system = ''

        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        if self.arch_type == 'qwen':
            self.patch_token = 1

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

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.use_fast = use_fast
        self.n_fast_images = n_fast_images
        self.fast_pool_size = fast_pool_size

        self.frame_contiguous_sample = frame_contiguous_sample

        # for visualization debug
        self.save_folder = './work_dirs/volume_debug/'
        self.cur_number = 0

        # exist_thr
        self.exist_thr = 8
        self.fast_token_after_question = fast_token_after_question
        if self.fast_token_after_question:
            assert self.use_fast

        print("volume res dataset, include {} items.".format(len(self.vid2metaid)))

    def __len__(self):
        return len(self.vid2metaid) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.vid2metaid:
            cur_len = 10000
            length_list.append(cur_len)
        return length_list

    def real_len(self):
        return len(self.vid2metaid)

    def json_file_preprocess(self, expression_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['volumes']

        metas = []
        anno_count = 0  # serve as anno_id
        vid2metaid = {}

        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]

            vid_slices = sorted(vid_express_data['slices'])
            vid_len = len(vid_slices)

            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['volume'] = vid_name
                meta['exp'] = exp_dict['exp']  # str

                meta['mask_file'] = exp_dict['mask_file']  # str
                meta['img_file'] = exp_dict['img_file']
                anno_count += 1
                meta['slices'] = vid_slices

                # print('check:', vid_len, len(meta['slices']), len(exp_dict['img_file']), len(exp_dict['mask_file']))
                # here we want to delete slices without mask file, which means these slices did not contain this class
                # !!!!
                meta = delete_slices(meta)
                # print('check:', vid_len, len(meta['slices']), len(exp_dict['img_file']), len(exp_dict['mask_file']))

                meta['exp_id'] = exp_id

                # meta['length'] = vid_len
                meta['length'] = len(meta['slices'])
                metas.append(meta)

                vid_name_exp = vid_name + '_' + exp_dict['exp']

                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name_exp] = []
                vid2metaid[vid_name_exp].append(len(metas) - 1)

        return vid2metaid, metas

    def decode_mask(self, volume_masks, image_size):
        ret_masks = []
        for object_masks in volume_masks:
            _object_masks = []

            for i_frame in object_masks:
                if i_frame != 'empty_mask':
                    m = Image.open(os.path.join(self.image_folder, i_frame))
                    m = np.asarray(m)
                    if len(m.shape) == 3:
                        m = m[:, :, 0]
                else:
                    m = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)

                m = cv2.resize(m, (image_size[1], image_size[0]))

                _object_masks.append(m)

            _object_masks = np.stack(_object_masks, axis=0)
            _object_masks = (_object_masks > 0).astype(np.uint8)

            ret_masks.append(_object_masks)
        _shape = ret_masks[0].shape
        for item in ret_masks:
            if item.shape != _shape:
                print([_ret_mask.shape for _ret_mask in ret_masks])
                return None
        ret_masks = np.stack(ret_masks, axis=0)  # (n_obj, n_slices, h, w)

        ret_masks = torch.from_numpy(ret_masks)
        # ret_masks = F.interpolate(ret_masks, size=(self.image_size // self.down_ratio,
        #                           self.image_size // self.down_ratio), mode='nearest')
        ret_masks = ret_masks.flatten(0, 1)
        return ret_masks

    def dataset_map_fn(self, data_dict, select_k=5):
        images = []

        len_slices = len(data_dict[0]['slices'])
        for objet_info in data_dict:
            assert len_slices == len(objet_info['slices'])

        # prepare images, random select k slices
        if len_slices > select_k + 1:
            # if self.frame_contiguous_sample and random.random() < 0.5:
            if self.frame_contiguous_sample:
                # do contiguous 3d_GVQA_data
                selected_start_frame = np.random.choice(len_slices - select_k, 1, replace=False)
                selected_frame_indexes = [selected_start_frame[0] + _i for _i in range(select_k)]
            else:
                selected_frame_indexes = np.random.choice(len_slices, select_k, replace=False)
        else:
            selected_frame_indexes = np.random.choice(len_slices, select_k, replace=True)
        selected_frame_indexes.sort()

        fast_volume_slices = None

        for selected_frame_index in selected_frame_indexes:
            frame_id = data_dict[0]['slices'][selected_frame_index]
            images.append(os.path.join(data_dict[0]['volume'], frame_id))

        # prepare text
        expressions = [object_info['exp'] for object_info in data_dict]

        if self.use_fast:
            text_dict = self.prepare_text(select_k, expressions, num_image_tokens=self.patch_token,
                                          n_fast_images=len(fast_volume_slices),)
        else:
            text_dict = self.prepare_text(select_k, expressions, num_image_tokens=self.patch_token)

        # prepare masks
        volume_masks = []
        for object_info in data_dict:
            frames_masks = object_info['mask_file']
            # print('check empty mask:', len(frames_masks), frames_masks, images, selected_frame_indexes)

            frames_masks_ = []
            for image_name in images:
                image_name = image_name.split('/')[-1][:-4]  # ignore '_mask' and '.png'
                find_mask_flag = False

                for each_frame_mask in frames_masks:
                    # print('check empty mask:', image_name, each_frame_mask.split('/')[-1][:len(image_name)])
                    if image_name == each_frame_mask.split('/')[-1][:len(image_name)]:
                        frames_masks_.append(copy.deepcopy(each_frame_mask))
                        find_mask_flag = True

                if find_mask_flag is False:
                    # did not find mask for this image, means this image did not contain this object
                    # append an empty_mask
                    frames_masks_.append('empty_mask')

            volume_masks.append(frames_masks_)

        fast_volume_masks = None

        # #  ! ! ! Revise conversations, without mask, There is no [SEG]
        # conversations = text_dict['conversation']
        # # print('len of conversations and volume_masks:', len(conversations), len(volume_masks))
        #
        # new_conversations = []
        # assert len(conversations) == len(volume_masks)
        # for i in range(len(volume_masks)):
        #     conversation = conversations[i]
        #     volume_mask = volume_masks[i]
        #     # print('conversation[output], volume_mask:', conversation['output'], volume_mask)
        #
        #     new_conversation = conversation.copy()
        #     all_empty_flag = all(element == 'empty_mask' for element in volume_mask)
        #     if all_empty_flag:
        #         new_conversation['output'] = 'There is no [SEG].'
        #
        #     new_conversations.append(new_conversation)

        # ret = {'images': images, 'volume_masks': volume_masks, 'conversation': new_conversations,
        #        'fast_images': fast_volume_slices, 'fast_volume_masks': fast_volume_masks}

        ret = {'images': images, 'volume_masks': volume_masks, 'conversation': text_dict['conversation'],
               'fast_images': fast_volume_slices, 'fast_volume_masks': fast_volume_masks}


        return ret

    def prepare_text(self, n_slices, expressions, num_image_tokens=256, n_fast_images=50):

        if self.use_fast and not self.fast_token_after_question:
            fast_frame_token_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_images * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}' + '\n'
        else:
            fast_frame_token_str = ''

        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'
        if self.fast_token_after_question:
            assert self.use_fast
            after_question_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_images * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}'
        else:
            after_question_str = ''

        questions = []
        answers = []
        for i, exp in enumerate(expressions):

            # the exp is a question
            if '?' in exp:
                questions.append(exp)
            else:
                exp = exp.replace('.', '').strip()
                question_template = random.choice(SEG_QUESTIONS)
                questions.append(question_template.format(class_name=exp.lower()))

            answers.append(random.choice(ANSWER_LIST))
        qa_list = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                frame_tokens = frame_token_str + '\n'
                # frame_tokens = '=' + ' '
                frame_tokens = frame_tokens * n_slices
                frame_tokens = frame_tokens.strip()
                frame_tokens = fast_frame_token_str + frame_tokens
                qa_list.append(
                    {'from': 'human', 'value': frame_tokens + question + after_question_str}
                )
            else:
                qa_list.append(
                    {'from': 'human', 'value': question + after_question_str}
                )
            qa_list.append(
                {'from': 'gpt', 'value': answer}
            )

        input = ''
        conversation = []

        for msg in qa_list:
            if msg['from'] == 'human':
                input += msg['value']
            elif msg['from'] == 'gpt':
                conversation.append({'input': input, 'output': msg['value']})
                input = ''
            else:
                raise NotImplementedError

        # add system information
        conversation[0].update({'system': self._system})
        return {'conversation': conversation}

    def __getitem__(self, index):

        index = index % self.real_len()

        selected_volume_objects = self.vid2metaid[self.volumes[index]]

        volume_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_volume_objects]

        # don't random, all classes input !
        # if len(volume_objects_infos) >= self.select_number:
        #     selected_indexes = np.random.choice(len(volume_objects_infos), self.select_number)
        #     volume_objects_infos = [volume_objects_infos[_idx] for _idx in selected_indexes]
        # else:
        #     selected_indexes = np.random.choice(len(volume_objects_infos), self.select_number, replace=True)
        #     volume_objects_infos = [volume_objects_infos[_idx] for _idx in selected_indexes] # list

        data_dict = self.dataset_map_fn(volume_objects_infos, select_k=self.sampled_slices)

        assert 'images' in data_dict.keys()
        pixel_values = []
        extra_pixel_values = []
        num_volume_tokens = None
        num_frame_tokens = None
        if data_dict.get('images', None) is not None:
            slices_files = data_dict['images']
            slices_files = [os.path.join(self.image_folder, frame_file) for frame_file in slices_files]
            for frame_path in slices_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size
                if self.extra_image_processor is not None:
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_pixel_values)

                if self.preprocessor is not None:
                    pass
                else:
                    frame_image = self.transformer(frame_image)
                pixel_values.append(frame_image)

            if self.preprocessor is not None:
                if self.arch_type == 'qwen':
                    _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                    num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                    num_slices = _data_dict['image_grid_thw'].shape[0]
                    num_volume_tokens = num_frame_tokens * num_slices
                elif self.arch_type == 'llava':
                    _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                else:
                    raise NotImplementedError
                data_dict.update(_data_dict)
            else:
                pixel_values = torch.stack(pixel_values, dim=0) # (n_f, 3, h, w)
                data_dict['pixel_values'] = pixel_values
            if self.extra_image_processor is not None:
                data_dict['g_pixel_values'] = extra_pixel_values

            # process and get masks
            masks = self.decode_mask(data_dict['volume_masks'], image_size=(ori_height, ori_width))
            if masks is None:
                return self.__getitem__(random.randint(0, self.real_len()))
            data_dict['masks'] = masks
        else:
            data_dict['pixel_values'] = torch.zeros(0, 3, self.image_size, self.image_size)
            data_dict['masks'] = None

        if num_volume_tokens is not None:
            assert self.patch_token == 1
            input_str = data_dict['conversation'][0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens)
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_volume_tokens
            data_dict['conversation'][0]['input'] = input_str

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = volume_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)

        # for fast branch
        if self.use_fast:
            fast_pixel_values = []
            slices_files = data_dict['fast_images']
            slices_files = [os.path.join(self.image_folder, frame_file) for frame_file in slices_files]
            for frame_path in slices_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size

                frame_image = self.transformer(frame_image)
                fast_pixel_values.append(frame_image)

            fast_pixel_values = torch.stack(fast_pixel_values, dim=0)  # (n_f, 3, h, w)
            data_dict['fast_pixel_values'] = fast_pixel_values

            # process and get masks
            masks = self.decode_mask(data_dict['fast_volume_masks'], image_size=(ori_height, ori_width))

            if masks is None:
                return self.__getitem__(random.randint(0, self.real_len()))

            data_dict['fast_exists'] = masks.to(dtype=torch.int).sum(dim=(-2, -1)).ge(self.exist_thr).unsqueeze(-1)

            del data_dict['fast_volume_masks']
        data_dict['type'] = 'volume'
        return data_dict