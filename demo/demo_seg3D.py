import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from demo.eval_utils.eval_seg3D import save_metrics_json

try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed; visualization is disabled.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="video Reasoning Segmentation for val set"
    )
    parser.add_argument("--val_folder", default='/data/linshan/Biomedparse/3D/CHAOS/',
                        help="Path to the val dataset")

    parser.add_argument("--val_json", default='meta_test.json',
                        help="Path to the meta json of val dataset")

    parser.add_argument(
        "--model_path", default="ByteDance/Sa2VA-4B", help="HF model name or path"
    )
    parser.add_argument(
        "--work-dir",
        default="./val_results/3D/CHAOS",
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


def visualize_and_save(pred_mask, frame_path, out_path):
    """
    Overlays the binary mask on the original frame and saves a PNG image to out_path.
    """
    if Visualizer is None:
        print("mmengine is not installed, skipping visualization.")
        return

    visualizer = Visualizer()
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image file: {frame_path}")
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors="g", alphas=0.4)
    visual_result = visualizer.get_image()
    cv2.imwrite(out_path, visual_result)


def main():
    cfg = parse_args()

    # 1. Load model and tokenizer
    print(f"Loading model from {cfg.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    model.eval()

    # 2. Read the "expressions.json" from val folder
    val_path = Path(cfg.val_folder)
    ann_file = os.path.join(cfg.val_folder, cfg.val_json)
    ann_file = Path(ann_file)

    if not ann_file.exists():
        raise FileNotFoundError(
            f"Could not find {ann_file}. Is the val folder correct?"
        )

    with open(ann_file, "r") as f:
        annotation_data = json.load(f)

    # The JSON structure is typically: {"videos": {video_id: {frames: [...], expressions: {...}}}}
    videos_info = annotation_data["volumes"]
    print(f"Found {len(videos_info)} volumes in {ann_file}")

    # 3. Prepare output directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    print(f"Results will be saved to: {cfg.work_dir}\n")

    # 4. Loop over each video, then each expression
    with torch.inference_mode():
        for video_id, vid_data in videos_info.items():
            # frames for this video
            vid_frames = sorted(vid_data["slices"])  # e.g., ["0001", "0002", ...]

            # Load all frames as PIL
            # {video_id}/{frame_name}
            frame_paths = [
                os.path.join(val_path, video_id, f"{fn}")
                for fn in vid_frames
            ]
            frames_pil = []
            for fp in frame_paths:
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"slice not found: {fp}")

                fp = Image.open(fp).convert("RGB")
                # if fp.size[:2] != (1024, 1024):
                #     fp = fp.resize((1024, 1024))

                frames_pil.append(fp)

            # Go through each expression in this video
            expressions_dict = vid_data["expressions"]
            for exp_id, exp_info in expressions_dict.items():
                out_dir = os.path.join(cfg.work_dir, video_id, exp_id)
                if os.path.exists(out_dir):
                    continue
                # We'll form a text prompt for the model
                # The default text is cfg.text, e.g., "<image>Please describe the video content."
                # Some code might append the expression, but you can adapt as needed.
                # For example:
                # text_prompt = f"<image> Describe: {exp_info['exp']}"
                # Or simply use cfg.text.
                text_prompt = f"<image>Please segment {exp_info['exp']}."
                print('text_prompt: ------', text_prompt)

                print(
                    f"Processing volume={video_id}, exp_id={exp_id} with {len(frames_pil)} slices."
                )

                # 5. Run the model for the entire video
                result = model.predict_forward(
                    video=frames_pil, text=text_prompt, tokenizer=tokenizer
                )

                # 6. Print or log the raw language model output
                prediction_text = result["prediction"]
                print(f"  Model output: {prediction_text}")

                # 7. Check for segmentations, overlay, and save
                if "[SEG]" in prediction_text and Visualizer is not None:
                    print('prediction_text:', prediction_text)
                    # The model might return multiple sets of masks,
                    # typically `result['prediction_masks']`
                    # Here we assume there's at least one set
                    seg_idx = 0
                    if "prediction_masks" not in result:
                        print(
                            "  prediction_masks missing from result; cannot visualize."
                        )
                        continue
                    pred_masks = result["prediction_masks"][seg_idx]

                    # Create output folder: {work_dir}/{video_id}/{exp_id}
                    # out_dir = os.path.join(cfg.work_dir, video_id, exp_id)
                    out_dir = os.path.join(cfg.work_dir, video_id, str(exp_info['exp']))
                    os.makedirs(out_dir, exist_ok=True)

                    # For each frame, overlay mask and save
                    for frame_idx, frame_name in enumerate(vid_frames):
                        pred_mask = pred_masks[frame_idx]
                        out_path = os.path.join(out_dir, f"{frame_name}")
                        mask = pred_mask.astype(np.float32)
                        mask = Image.fromarray(mask * 255).convert("L")
                        mask.save(out_path)

                else:
                    print(
                        "  No segmentation or Visualizer unavailable; skipping mask overlay."
                    )

    print("\nDone processing all videos and expressions!")

    save_metrics_json(cfg.val_folder, cfg.work_dir, cfg.val_json)


if __name__ == "__main__":
    main()