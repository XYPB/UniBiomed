import torch
import json
import os
import datetime
from copy import deepcopy
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from transformers import (AutoModel, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          GenerationConfig)

import argparse

from preprocess_eval_datasets import (
    parse_pmc_vqa_to_multi_choice_conversations,
    parse_mecovqa_json_to_conversations,
    parse_rad_vqa_json_to_conversations,
    parse_mecog_json,
    parse_medxpertqa_json,
    parse_omnimedvqa_jsons,
    parse_medsynth_omni_json,
    parse_pvqa_to_conversations,
    parse_slake_json_to_conversations,
    parse_imageclef_jsonl_to_conversations,
    parse_mecovqa_region_json_to_conversations,
    parse_mecovqa_region_yn_json_to_conversations,
    parse_medsynth_no_region_json_to_conversations,
    parse_mecovqa_region_yn_no_region_json_to_conversations,
)

parser = argparse.ArgumentParser(description="Evaluate VLLM models on MeCoVQA dataset")
parser.add_argument("--dataset", type=str, default="MeCoVQA", help="Dataset to evaluate on (default: MeCoVQA)")
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate (default: 10)")
parser.add_argument("--temperature", type=float, default=0, help="Temperature for model inference (default: 0)")
parser.add_argument("--bbox_coord", action='store_true', help="Use bounding box coordinates for models that support it (default: False)")
parser.add_argument("--side_by_side", action='store_true', help="Use side-by-side mask visualization for models that support it (default: False)")
parser.add_argument("--skip_region", action='store_true', help="Skip region highlighting in the image (default: False)")


torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)

def highlight_region(image, mask, alpha=0.25):
    """
    Highlight a specific mask in the image by drawing a transparent overlay around it.
    """
    # Both image and mask should be PIL Images
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2_mask = np.array(mask)

    # Ensure mask is binary
    if cv2_mask.ndim == 3:
        cv2_mask = cv2_mask[:, :, 0]
    cv2_mask = (cv2_mask > 0).astype(np.uint8) * 255
    # Create a colored overlay
    color = (0, 0, 255)
    overlay = np.zeros_like(cv2_image, dtype=np.uint8)
    overlay[cv2_mask > 0] = color
    # Blend the overlay with the original image
    highlighted_image = cv2.addWeighted(cv2_image, 1 - alpha, overlay, alpha, 0)
    return Image.fromarray(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))

def highlight_region_bbox(image, mask, width=5, color=(0, 0, 255)):
    """
    Highlight a bounding box in the image by drawing a rectangle around it.
    """
    # Convert image to numpy array
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert mask to bbox coordinates
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8) * 255
    x1 = np.min(np.where(mask > 0)[1])
    x2 = np.max(np.where(mask > 0)[1])
    y1 = np.min(np.where(mask > 0)[0])
    y2 = np.max(np.where(mask > 0)[0])

    # Draw the rectangle on the image
    cv2.rectangle(cv2_image, (x1, y1), (x2, y2), color, width)

    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def mask_side_by_side(image, mask):
    """
    Combine the original image and the mask side by side.
    """
    # Convert image and mask to numpy arrays
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2_mask = cv2.cvtColor(np.array(mask), cv2.COLOR_GRAY2BGR)

    # Ensure mask is binary
    # if cv2_mask.ndim == 3:
    #     cv2_mask = cv2_mask[:, :, 0]
    cv2_mask = (cv2_mask > 0).astype(np.uint8) * 255

    # Create a side-by-side image
    combined_image = np.hstack((cv2_image, cv2_mask))
    
    return Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))

def normalize_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_box

def mask_as_bbox(mask):
    image_width, image_height = mask.size
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8) * 255
    x1 = np.min(np.where(mask > 0)[1])
    x2 = np.max(np.where(mask > 0)[1])
    y1 = np.min(np.where(mask > 0)[0])
    y2 = np.max(np.where(mask > 0)[0])
    bbox = (x1, y1, x2, y2)
    return normalize_coordinates(bbox, image_width, image_height)

def save_outputs_to_json(outputs, filename, output_dir="./runs/output", model_info=None):
    """
    Save model outputs to a JSON file.
    
    Args:
        outputs (list): List of model outputs
        filename (str): Name of the output JSON file
        output_dir (str): Directory to save the outputs
        model_info (dict, optional): Additional model information to include
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to the output file
    output_path = os.path.join(output_dir, filename)
    
    # Create a results dictionary with metadata
    results = {
        "outputs": outputs,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(outputs)
    }
    
    # Add model info if provided
    if model_info:
        results.update({"model_info": model_info})
    
    # Save the outputs
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Outputs saved to {output_path}")
    return output_path


def eval_unibiomed(conversations, gts):
    model = AutoModel.from_pretrained(
            'Luffy503/UniBiomed',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        'Luffy503/UniBiomed',
        trust_remote_code=True,
    )
    outputs = []

    for idx, messages in tqdm(enumerate(conversations), total=len(conversations), desc="Evaluating UniBiomed"):
        image_path = messages[1]['content'][1]['image']
        image = Image.open(image_path).convert('RGB')

        question = messages[1]['content'][0]['text']
        if "<image>" not in question:
            question = "<image>\n" + question


        with torch.inference_mode():
            data_dict = {
                "image": image,
                "text": question
            }
            
            pred_dict = model.predict_forward(**data_dict, tokenizer=tokenizer)
            # text description
            prediction = pred_dict['prediction']
            # segmentation mask
            if 'prediction_masks' in pred_dict and len(pred_dict['prediction_masks']) > 0:
                mask = pred_dict['prediction_masks'][0][0]
                mask = Image.fromarray((mask*255).astype('uint8'))
            
                image_suffix = image_path.split('.')[-1]
                mask_path = image_path.replace(f".{image_suffix}", "unibiomed_mask.png")
                mask.convert("L").save(mask_path)
            else:
                mask_path = ''

            output = {
                "id": image_path,
                "input": messages[1]["content"][0]['text'],
                "output": prediction,
                "mask": mask_path,
                "gt": gts[idx] if idx < len(gts) else None
            }
            outputs.append(output)
    return outputs


if __name__ == "__main__":
    args = parser.parse_args()
    # Create a timestamped directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    size = "full" if args.num_samples <= 0 else args.num_samples
    output_dir = os.path.join("./runs/output", f"eval_{timestamp}_{args.dataset}_{size}_unibiomed")
    if args.skip_region:
        output_dir += "_skip_region"
    if args.bbox_coord:
        output_dir += "_bbox_coord"
    if args.side_by_side:
        output_dir += "_side_by_side"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.dataset == "PMC-VQA":
        data_path = "./data/PMC-VQA/test_2.csv"
        _, _, conversations, gts = parse_pmc_vqa_to_multi_choice_conversations(data_path)
    elif args.dataset == "MeCoVQA":
        data_path = 'data/MeCoVQA/test/MeCoVQA_Complex_VQA_test.json'
        conversations, gts = parse_mecovqa_json_to_conversations(data_path)
    elif args.dataset == "MeCoVQA_region":
        data_path = 'data/MeCoVQA/test/MeCoVQA_Region_VQA_test.json'
        conversations, gts = parse_mecovqa_region_json_to_conversations(data_path)
    elif args.dataset == "MeCoVQA_region_yn":
        data_path = 'data/MeCoVQA/test/MeCoVQA_Region_Closed_VQA_test.json'
        conversations, gts = parse_mecovqa_region_yn_json_to_conversations(data_path)
    elif args.dataset == "MeCoVQA_region_yn_hard":
        data_path = 'data/MeCoVQA/test/MeCoVQA_Region_Closed_VQA_test_hard.json'
        conversations, gts = parse_mecovqa_region_yn_json_to_conversations(data_path)
    elif args.dataset == "MeCoVQA_region_yn_clean_no_region":
        data_path = 'data/MeCoVQA/test/MeCoVQA_Region_Closed_VQA_test_clean_no_region.json'
        conversations, gts = parse_mecovqa_region_yn_no_region_json_to_conversations(data_path)
    elif args.dataset == "medsynth_no_region":
        data_path = 'data/MeCoVQA/test/medvqa_synth_stage2_regionvqa_rl_test_maskless.json'
        conversations, gts = parse_medsynth_no_region_json_to_conversations(data_path)
    elif args.dataset == "VQA-RAD":
        data_path = './data/VQA_RAD/VQA_RAD Dataset Public.json'
        conversations, gts = parse_rad_vqa_json_to_conversations(data_path)
    elif args.dataset == "OmniMedVQA":
        # data_path = './data/OmniMedVQA/QA_information/Open-access/'
        data_path = "./data/OmniMedVQA/QA_information/Open-access/omnimedvqa_test_3k.jsonl"
        conversations, gts = parse_omnimedvqa_jsons(data_path)
    elif args.dataset == "mecovqa_g":
        data_path = "/data/yuexi/datasets/MeCoVQA/test/MeCoVQA_Grounding_test_adapted.jsonl"
        conversations, gts = parse_mecog_json(data_path)
    elif args.dataset == "PVQA":
        data_path = './data/pvqa/qas/test_vqa.pkl'
        conversations, gts = parse_pvqa_to_conversations(data_path)
    elif args.dataset == "SLAKE":
        data_path = './data/SLAKE/test.json'
        conversations, gts = parse_slake_json_to_conversations(data_path)
    elif args.dataset == "imageclef":
        data_path = 'data/ImageCLEF-2019/VQAMed2019Test/VQAMed2019_Test_Questions_w_Ref_Answers.jsonl'
        conversations, gts = parse_imageclef_jsonl_to_conversations(data_path)
    elif args.dataset == "MedXpertQA":
        data_path = "data/MedXpertQA/MM/test.jsonl"
        conversations, gts = parse_medxpertqa_json(data_path)
    elif args.dataset == 'MedSynth_omni_test':
        data_path = "data/SA-Med2D/medsynth_omni_test_1k_test_qa.jsonl"
        conversations, gts = parse_medsynth_omni_json(data_path)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets are: PMC-VQA, MeCoVQA, VQA-RAD, OmniMedVQA.")
    
    # Number of samples to evaluate
    num_samples = args.num_samples if args.num_samples > 0 else len(conversations)
    # Save evaluation configuration

    model_name = 'Luffy503/UniBiomed'

    model_config = {
        "name": model_name,
        "processing": "Sequential"
    }
    config = {
        "timestamp": timestamp,
        "dataset": data_path,
        "num_samples": num_samples,
        "models": model_config
    }
    config_path = os.path.join(output_dir, "eval_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Evaluate UniBiomed
    print(f"Evaluating UniBiomed on {num_samples} samples...")
    UniBiomed_outputs = eval_unibiomed(deepcopy(conversations)[:num_samples], gts[:num_samples])
    UniBiomed_model_info = {
        "model_name": model_name,
        "model_type": "Image-Text-to-Text",
        "batch_size": "N/A (Sequential processing)"
    }
    UniBiomed_output_path = save_outputs_to_json(
        UniBiomed_outputs,
        "UniBiomed_outputs.json",
        output_dir,
        UniBiomed_model_info
    )
    
    print(f"\nAll outputs saved to: {output_dir}")

