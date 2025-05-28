import argparse
import json
import os

import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from demo.eval_utils.eval_seg2D import save_metrics_json


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed; visualization is disabled.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="video Reasoning Segmentation for val set"
    )
    parser.add_argument("--val_folder", default='/data/linshan/Biomedparse/CHAOS',
                        help="Path to the val dataset")

    parser.add_argument("--val_json", default='test.json',
                        help="Path to the meta json of val dataset")

    parser.add_argument(
        "--model_path", default="./save_hf", help="HF model name or path"
    )
    parser.add_argument(
        "--work-dir",
        default="./val_results/2D/CHAOS",
        help="Directory to save segmented results",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="<image>Please describe the video content.",
        help="Prompt text sent to the model.",
    )
    args = parser.parse_args()
    return args


def main():
    cfg = parse_args()

    # 1. Load model and tokenizer
    print(f"Loading model from {cfg.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    # 2. Read the "expressions.json" from val folder
    json_file = os.path.join(cfg.val_folder, cfg.val_json)
    with open(json_file, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']

    # 3. Prepare output directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    print(f"Results will be saved to: {cfg.work_dir}\n")

    # 4. Loop over each image, then each expression
    from tqdm import tqdm

    with torch.inference_mode():
        for i in tqdm(range(len(annotations))):
            each_item = annotations[i]

            image_file = os.path.join(cfg.val_folder, each_item['split'], each_item['file_name'])
            image = Image.open(image_file).convert('RGB')

            # if image.size[:2] != (1024, 1024):
            #     image = image.resize((1024, 1024))

            text_prompt = f"<image>Please segment {each_item['sentences'][0]['sent']}."

            data_batch = {}
            data_batch['image'] = image
            data_batch['text'] = text_prompt

            pred_mask = model.predict_forward(**data_batch, tokenizer=tokenizer)['prediction_masks']

            # make save path
            os.makedirs(os.path.join(cfg.work_dir, each_item['split']), exist_ok=True)
            out_path = os.path.join(cfg.work_dir, each_item['split'], f"{each_item['mask_file']}")

            pred_mask = pred_mask[0][0]
            mask = pred_mask.astype(np.float32)
            print(mask.shape)
            mask = Image.fromarray(mask * 255).convert("L")

            mask.save(out_path)

    save_metrics_json(cfg.val_folder, cfg.work_dir, cfg.val_json)


if __name__ == "__main__":
    main()