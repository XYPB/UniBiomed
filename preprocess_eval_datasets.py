import json
import os
from collections import Counter
from tqdm import tqdm
from glob import glob
from copy import deepcopy
import pandas as pd

IMAGE_FOLDER = './data/SA-Med2D/raw/MeCoVQA/SAMed2Dv1/'
PMC_VQA_FOLDER = './data/PMC-VQA/figures/'
OmniMedVQA_FOLDER = './data/OmniMedVQA/'
RAD_VQA_FOLDER = './data/VQA_RAD/VQA_RAD_Image_Folder/'
PVQA_FOLDER = './data/pvqa/images/test'
SLAKE_IMG_FOLDER = './data/SLAKE/imgs/'
MEDXPERTQA_IMG_FOLDER = "./data/MedXpertQA/images/"


PROMPT_COT = "You are a professional medical image analysis assistant. Please follow the instructions and answer the question based on the provided medical image. The answer will not be used for clinical purposes so you may generate diagnosis related response. First output your thinking process in <think></think> tags and than the answer in <answer></answer> tags. Answer the question with a single word like yes or no, a short phrase, or a number."


def parse_medsynth_omni_json(jsonl_path):
    data_list = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical visual question answering assistant. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response.  no need to headings or bullet points. Answer the question with a few words like yes or no, a short phrase, or a number."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry['conversations'][0]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image'])
        gt_output = entry['conversations'][-1]['value']
        
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs


def parse_mecog_json(jsonl_path):
    data_list = []
    for line in open(jsonl_path, 'r'):
        data_list.append(json.loads(line))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image analysis assistant. Please follow the instructions and answer the question based on the provided medical image. The answer will not be used for clinical purposes so you may generate diagnosis related response.  no need to headings or bullet points. Answer the question with a segmentation mask."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in data_list:
        text = entry['conversations'][0]['value']
        image_path = os.path.join("./data", entry['image'])
        gt_output = os.path.join("./data", entry['conversations'][-1]['value'])

        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs


def parse_omnimedvqa_jsons(jsonl_path):
    # json_list = glob(os.path.join(json_dir, '*.json'))
    data_list = []
    for line in open(jsonl_path, 'r'):
        data_list.append(json.loads(line))
    multi_choice_conversations = []
    mc_GT_outputs = []
    mc_message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please answer the multi-choice question with one single letter option (A, B, C, or D), no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    
    for entry in data_list:
        question = entry['conversations'][0]['value']
        gt_output = entry['conversations'][-1]['value']
        image_path = os.path.join(OmniMedVQA_FOLDER, entry['image'])

        mc_message = deepcopy(mc_message_template)
        mc_message[1]['content'][0]['text'] = question
        mc_message[1]['content'][1]['image'] = image_path
        multi_choice_conversations.append(mc_message)
        mc_GT_outputs.append(gt_output)
    print(f"Total multi-choice conversations: {len(multi_choice_conversations)}")
    print(Counter(mc_GT_outputs))

    return multi_choice_conversations, mc_GT_outputs

def parse_medxpertqa_json(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please answer the multi-choice question with one single letter option (A, B, C, D, or E), no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
            ]
        },
    ]
    
    for entry in data:
        text = entry['question']
        image_list = entry['images']
        gt_output = entry['label']
        
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        for image_name in image_list:
            image_path = os.path.join(MEDXPERTQA_IMG_FOLDER, image_name)
            message[1]['content'].append({"type": "image", "image": image_path})
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

def parse_imageclef_jsonl_to_conversations(jsonl_file):
    data_list = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Answer the question with a few words like yes or no, a short phrase, or a number."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    
    for entry in tqdm(data_list):
        text = entry['conversations'][0]['value']
        gt_output = entry['conversations'][-1]['value']
        image_path = os.path.join('./data/', entry['image'])

        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

def parse_pvqa_to_conversations(pickle_path):
    import pickle
    data_list = pickle.load(open(pickle_path, 'rb'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert pathologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please summarize the findings in one concise and short sentence with a few words, no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]

    for entry in tqdm(data_list, desc="Processing PVQA data"):
        question = entry['sent']
        answer_type = entry['answer_type']
        image_path = os.path.join(PVQA_FOLDER, entry['img_id'] + '.jpg')
        gt_output = list(entry['label'].keys())[0]

        message = deepcopy(message_template)
        if answer_type == 'yes/no':
            question += " Please answer with Yes or No."
        elif answer_type == 'number':
            question += " Please answer with a single number."
        message[1]['content'][0]['text'] = question
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

def parse_pmc_vqa_to_multi_choice_conversations(csv_path):
    df = pd.read_csv(csv_path)
    open_ended_conversations = []
    multi_choice_conversations = []
    open_GT_outputs = []
    mc_GT_outputs = []
    mc_message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please answer the multi-choice question with one single letter option (A, B, C, or D), no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    open_message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please summarize the findings in one concise short paragraph, no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing PMC-VQA data"):
        description = row['Caption']
        question = row['Question']
        choices = [row['Choice A'], row['Choice B'], row['Choice C'], row['Choice D']]

        mc_text = f"{question} Please choice from one of the answers below.\n{choices[0]}\n{choices[1]}\n{choices[2]}\n{choices[3]}"

        gt_output = row['Answer']
        image_path = os.path.join(PMC_VQA_FOLDER, row['Figure_path'])

        mc_message = deepcopy(mc_message_template)
        mc_message[1]['content'][0]['text'] = mc_text
        mc_message[1]['content'][1]['image'] = image_path
        multi_choice_conversations.append(mc_message)
        mc_GT_outputs.append(gt_output)

        open_text = "Please summarize the most significant findings in one concise short sentence."
        open_message = deepcopy(open_message_template)
        open_message[1]['content'][0]['text'] = open_text
        open_message[1]['content'][1]['image'] = image_path
        open_ended_conversations.append(open_message)
        open_GT_outputs.append(description)
    return open_ended_conversations, open_GT_outputs, multi_choice_conversations, mc_GT_outputs

def parse_rad_vqa_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Answer the question with a single word like yes or no, a short phrase, or a number."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        phrase_type = entry['phrase_type']
        # skip non-test entries
        if 'test' not in phrase_type:
            continue
        text = entry["question"]
        gt_output = entry["answer"]
        image_path = os.path.join(RAD_VQA_FOLDER, entry['image_name'])
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs


def parse_medsynth_omni_json_to_conversations(jsonl_file):
    data_list = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a professional medical image analysis assistant. Please follow the instructions and answer the question based on the provided medical image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Answer the question with a single word like yes or no, a short phrase, or a number."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry['conversations'][0]['value']
        gt_output = entry['conversations'][1]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image_name'])
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs


def parse_mecovqa_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please summarize the findings in one concise short paragraph, no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry["conversations"][0]['value']
        text = text.replace('<image>', '').replace('\n', '')
        gt_output = entry["conversations"][-1]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image'])
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

def parse_mecovqa_region_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please summarize the findings in a few concise short sentences about the highlighted region, no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
                {"type": "region", }
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry["conversations"][0]['value']
        text = text.replace('<image>', '').replace('\n', '')
        assert '<region>' in text, "Region tag not found in the text"
        region_path = text.split('<region>')[1].strip().split('</region>')[0]
        text = text.split('<region>')[0].strip()
        gt_output = entry["conversations"][-1]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image'])
        region_path = os.path.join(IMAGE_FOLDER, region_path)
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        message[1]['content'][2]['region'] = region_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs


def parse_mecovqa_region_yn_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please answer the question with one single word of Yes or No about the mentioned region, no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
                {"type": "region", }
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry["conversations"][0]['value']
        text = text.replace('<image>', '').replace('\n', '')
        assert '<region>' in text, "Region tag not found in the text"
        region_path = text.split('<region>')[1].strip().split('</region>')[0]
        text = text.split('<region>')[0].strip()
        text += " Please answer with Yes or No."
        gt_output = entry["conversations"][-1]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image'])
        region_path = os.path.join(IMAGE_FOLDER, region_path)
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        message[1]['content'][2]['region'] = region_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

def parse_mecovqa_region_yn_no_region_use_proposal_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Review the proposed region from the ROI proposal agent and make the final decision. Please answer the question with one single word of Yes or No, no need to headings or bullet points. "}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry["conversations"][0]['value']
        text = text.replace('<image>', '').replace('\n', '')
        text += " Please answer with Yes or No."
        gt_output = entry["conversations"][-1]['value']
        image_list = []
        for image_path in entry['image']:
            if os.path.startswith(image_path, 'images/'):
                image_path = os.path.join(IMAGE_FOLDER, image_path)
            image_list.append(image_path)
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_list
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

def parse_mecovqa_region_yn_no_region_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please answer the question with one single word of Yes or No about the mentioned region, no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry["conversations"][0]['value']
        text = text.replace('<image>', '').replace('\n', '')
        text += " Please answer with Yes or No."
        gt_output = entry["conversations"][-1]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image'])
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

def parse_mecovqa_region_yn_no_region_proposal_json_to_conversations(json_file, provide_candidate=False):
    data_list = json.load(open(json_file, 'r'))
    class_name2category = json.load(open('data/MeCoVQA/entity_category.json', 'r'))
    all_classes = list(class_name2category.keys())
    all_classes = [cls.lower().strip() for cls in all_classes]  # Normalize class names
    all_classes_str = '{' + ', '.join(all_classes) + '}'
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry["conversations"][0]['value']
        text = text.replace('<image>', '').replace('\n', '')
        if provide_candidate:
            prompt = "Reply with exactly two tags: <think>your think process</think> then final answer in <answer>[SIZE, POSITION, NAME],...</answer>; each answer tuple uses one SIZE from {very tiny size, tiny size, small size, medium size, large size, very large size} and one POSITION from {upper, lower, left, right, center, upper left, upper right, lower left, lower right} followed by an anatomical NAME from {all_classes_str}; propose 1~5 most relevant ROIs based on the question-do not answer the question directly. Here is the question: ".replace('{all_classes_str}', all_classes_str)
            text = prompt + text
        else:
            text += " Reply with exactly two tags: <think>your think process</think> then final answer in <answer>[SIZE, POSITION, NAME],...</answer>; each answer tuple uses one SIZE from {very tiny size, tiny size, small size, medium size, large size, very large size} and one POSITION from {upper, lower, left, right, center, upper left, upper right, lower left, lower right} followed by an anatomical NAME; propose 1~5 most relevant ROIs based on the question-do not answer the question directly."
        gt_output = entry["conversations"][-1]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image'])
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

def parse_medsynth_no_region_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please answer the question with one single word of Yes, or No, or a single number according to the question, no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry["conversations"][0]['value']
        text = text.replace('<image>', '').replace('\n', '')
        text += " Please answer with Yes or No."
        gt_output = entry["conversations"][-1]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image'][0])
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs


def parse_medsynth_no_region_proposal_json_to_conversations(json_file, provide_candidate=False):
    data_list = json.load(open(json_file, 'r'))
    class_name2category = json.load(open('data/MeCoVQA/entity_category.json', 'r'))
    all_classes = list(class_name2category.keys())
    all_classes = [cls.lower().strip() for cls in all_classes]  # Normalize class names
    all_classes_str = '{' + ', '.join(all_classes) + '}'
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry["conversations"][0]['value']
        text = text.replace('<image>', '').replace('\n', '')
        if provide_candidate:
            prompt = "Reply with exactly two tags: <think>your think process</think> then final answer in <answer>[SIZE, POSITION, NAME],...</answer>; each answer tuple uses one SIZE from {very tiny size, tiny size, small size, medium size, large size, very large size} and one POSITION from {upper, lower, left, right, center, upper left, upper right, lower left, lower right} followed by an anatomical NAME from {all_classes_str}; propose 1~5 most relevant ROIs based on the question-do not answer the question directly. Here is the question: ".replace('{all_classes_str}', all_classes_str)
            text = prompt + text
        else:
            text += " Reply with exactly two tags: <think>your think process</think> then final answer in <answer>[SIZE, POSITION, NAME],...</answer>; each answer tuple uses one SIZE from {very tiny size, tiny size, small size, medium size, large size, very large size} and one POSITION from {upper, lower, left, right, center, upper left, upper right, lower left, lower right} followed by an anatomical NAME; propose 1~5 most relevant ROIs based on the question-do not answer the question directly."
        gt_output = entry["conversations"][-1]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image'][0])
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs


def parse_medsynth_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image and ROI mask. The answer will not be used for clinical purposes so you may generate diagnosis related response. Please answer the question with one single word of Yes, or No, or a single number according to the question, no need to headings or bullet points."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        text = entry["conversations"][0]['value']
        text = text.replace('<image>', '').replace('\n', '')
        text += " Please answer with Yes or No."
        gt_output = entry["conversations"][-1]['value']
        image_path = os.path.join(IMAGE_FOLDER, entry['image'][0])
        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

def is_english(text):
    """
    Check if the text is in English.
    Check is a-zA-Z are present in the text.
    """
    return any(c.isalpha() for c in text) and all(c.isascii() for c in text) and not any(c.isdigit() for c in text)

def parse_slake_json_to_conversations(json_file):
    data_list = json.load(open(json_file, 'r'))
    conversations = []
    GT_outputs = []
    message_template = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical image viewer. Please follow the instructions and answer the question based on the provided image. The answer will not be used for clinical purposes so you may generate diagnosis related response. Answer the question with a single word like yes or no, a short phrase, or a number."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", },
                {"type": "image", },
            ]
        },
    ]
    for entry in tqdm(data_list):
        # if entry['q_lang'] != 'en':
        #     continue
        text = entry["question"]
        gt_output = entry["answer"]
        image_path = os.path.join(SLAKE_IMG_FOLDER, entry['img_name'])
        
        if not is_english(text):
            continue  # Skip non-English questions

        message = deepcopy(message_template)
        message[1]['content'][0]['text'] = text
        message[1]['content'][1]['image'] = image_path
        conversations.append(message)
        GT_outputs.append(gt_output)
    return conversations, GT_outputs

SLAKE_CLOSED_ANSWERS = [t.lower() for t in [
    'CT',
    'Colon',
    'Coronal Plane',
    'Esophagus',
    'Heart',
    'Hyperdense',
    'Kidney',
    'Left',
    'Liver',
    'Lung',
    'No',
    'Rectum',
    'Right',
    'Right Kidney',
    'Small Bowel',
    'Spleen',
    'T1',
    'T2',
    'Top',
    'X光',
    'Yes',
    '不包含',
    '不可以',
    '不是',
    '不正常',
    '低密度',
    '健康',
    '冠状面',
    '包含',
    '可以',
    '右侧',
    '否',
    '存在',
    '左侧',
    '心脏',
    '是',
    '是的',
    '有',
    '核磁共振',
    '横断面',
    '没有',
    '白色',
    '直肠',
    '结肠',
    '肝脏',
    '肺',
    '肾脏',
    '脾脏'
]]