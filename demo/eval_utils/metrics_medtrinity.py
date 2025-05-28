import os
import json
import argparse
import random

from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from pycocoevalcap.eval import COCOEvalCap,  Bleu, Meteor, Rouge, Cider, Spice, PTBTokenizer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import time
import copy
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--split", default='test', help="Evaluation split, options are 'val', 'test'")
    parser.add_argument("--root", default='./data/Biomed/RadGenome', help="data root")
    parser.add_argument("--prediction_dir_path", default='./val_results/Grounded_Report_Generation/RadGenome',
                        help="The path where the inference results are stored.")
    parser.add_argument("--gt_json_path", required=False, default="test.json",
                        help="The path containing GranD-f evaluation annotations.")

    args = parser.parse_args()

    return args


# Load pre-trained model tokenizer and model for evaluation
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


class COCOEvalCap_GRG(COCOEvalCap):
    def __init__(self, coco, cocoRes):
        super().__init__(coco, cocoRes)

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            res[imgId] = self.cocoRes.imgToAnns[imgId]
            temp = self.coco.imgToAnns[imgId]
            # gts[imgId] = self.coco.imgToAnns[imgId]

            if len(temp) == len(res[imgId]) and temp[0]['caption'] is not None:
                temp_list = []
                for idx, item in enumerate(temp):
                    new_item = res[imgId][idx].copy()
                    new_item['caption'] = item['caption']
                    del new_item['labels']
                    temp_list.append(new_item)

                gts[imgId] = temp_list

                print('gts[imgId]:', gts[imgId])
                print('res[imgId]:', res[imgId])
                print('\n')
            else:
                del res[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print
            'computing %s score...' % (scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print
                    "%s: %0.3f" % (m, sc)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print
                "%s: %0.3f" % (method, score)
        self.setEvalImgs()


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    # Use the mean of the last hidden states as sentence embedding
    sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0).detach().numpy()

    return sentence_embedding


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)

    return iou


def bbox_to_x1y1x2y2(bbox):
    x1, y1, w, h = bbox
    bbox = [x1, y1, x1 + w, y1 + h]

    return bbox


def compute_miou(pred_masks, gt_masks):
    # Computing mIoU between predicted masks and ground truth masks
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    # One-to-one pairing and mean IoU calculation
    paired_iou = []
    while iou_matrix.size > 0 and np.max(iou_matrix) > 0:
        max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        paired_iou.append(iou_matrix[max_iou_idx])
        iou_matrix = np.delete(iou_matrix, max_iou_idx[0], axis=0)
        iou_matrix = np.delete(iou_matrix, max_iou_idx[1], axis=1)

    return np.mean(paired_iou) if paired_iou else 0.0


def compute_dice(pred_mask, gt_mask):
    """Dice"""
    # pred_mask, gt_mask = pred_mask[0], gt_mask[0]

    intersection = np.sum(pred_mask * gt_mask)
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)

    # 计算 Dice 系数
    if pred_sum + gt_sum == 0:
        return 1.0
    return 2.0 * intersection / (pred_sum + gt_sum)


def compute_mean_dice(pred_masks, gt_masks):
    # 计算预测掩码和真实掩码之间的平均 Dice 系数
    dice_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            dice_matrix[i, j] = compute_dice(pred_mask, gt_mask)

    # 一对一配对和平均 Dice 计算
    paired_dice = []
    while dice_matrix.size > 0 and np.max(dice_matrix) > 0:
        max_dice_idx = np.unravel_index(np.argmax(dice_matrix, axis=None), dice_matrix.shape)
        paired_dice.append(dice_matrix[max_dice_idx])
        dice_matrix = np.delete(dice_matrix, max_dice_idx[0], axis=0)
        dice_matrix = np.delete(dice_matrix, max_dice_idx[1], axis=1)

    return np.mean(paired_dice) if paired_dice else 0.0


def evaluate_mask_mdice(coco_gt, image_ids, pred_save_path, root, split):
    # Load predictions
    coco_dt = coco_gt.loadRes(pred_save_path)

    dices = []
    for image_id in tqdm(image_ids):
        # Getting ground truth masks
        matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == image_id]
        ann_ids = [ann['id'] for ann in matching_anns]

        gt_anns = coco_gt.loadAnns(ann_ids)
        # print(gt_anns[0].keys())
        gt_masks = []
        for ann in gt_anns:
            mask_file = ann['mask_file']
            mask = Image.open(os.path.join(root, split+'_mask', mask_file))
            mask = np.asarray(mask)
            mask = (mask > 0).astype(np.uint8)
            gt_masks.append(mask)

        # gt_masks = [maskUtils.decode(ann['segmentation']) for ann in gt_anns if 'segmentation' in ann]

        # Getting predicted masks
        matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == image_id]
        dt_ann_ids = [ann['id'] for ann in matching_anns]
        pred_anns = coco_dt.loadAnns(dt_ann_ids)
        pred_masks = [maskUtils.decode(ann['segmentation']) for ann in pred_anns if 'segmentation' in ann]

        # print('gt_masks:', len(gt_masks), gt_masks[0].shape, np.unique(gt_masks[0]))
        # print('pred_masks:', len(pred_masks), pred_masks[0].shape, np.unique(pred_masks[0]))

        # Compute and save the dice for the current image
        dices.append(compute_mean_dice(pred_masks, gt_masks))

    # Report mean IoU across all images
    mean_dice = np.mean(dices) if dices else 0.0  # If list is empty, return 0.0

    return mean_dice


def compute_iou_matrix(pred_masks, gt_masks):
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    return iou_matrix


def text_similarity_bert(str1, str2):
    emb1 = get_bert_embedding(str1)
    emb2 = get_bert_embedding(str2)

    return cosine_similarity([emb1], [emb2])[0, 0]


def find_best_matches(gt_anns, gt_labels, dt_anns, dt_labels, iou_threshold, text_sim_threshold, vectorizer=None):
    best_matches = []

    # Compute pair - wise IoU
    pred_masks = [maskUtils.decode(ann['segmentation']) for ann in dt_anns]
    gt_masks = [maskUtils.decode(ann['segmentation']) for ann in gt_anns]
    ious = compute_iou_matrix(gt_masks, pred_masks)

    text_sims = np.zeros((len(gt_labels), len(dt_labels)))

    for i, gt_label in enumerate(gt_labels):
        for j, dt_label in enumerate(dt_labels):
            text_sims[i, j] = text_similarity_bert(gt_label, dt_label)

    # Find one-to-one matches satisfying both IoU and text similarity thresholds
    while ious.size > 0:
        max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[max_iou_idx] < iou_threshold or text_sims[max_iou_idx] < text_sim_threshold:
            break  # No admissible pair found

        best_matches.append(max_iou_idx)

        # Remove selected annotations from consideration
        ious[max_iou_idx[0], :] = 0
        ious[:, max_iou_idx[1]] = 0
        text_sims[max_iou_idx[0], :] = 0
        text_sims[:, max_iou_idx[1]] = 0

    return best_matches  # List of index pairs [(gt_idx, dt_idx), ...]


def evaluate_recall_with_mapping(coco_gt, coco_cap_gt, image_ids, pred_save_path, cap_pred_save_path, iou_threshold=0.5,
                                 text_sim_threshold=0.5):
    coco_dt = coco_gt.loadRes(pred_save_path)
    coco_cap_dt = coco_cap_gt.loadRes(cap_pred_save_path)

    true_positives = 0
    actual_positives = 0

    for image_id in tqdm(image_ids):
        try:
            # gt_ann_ids = coco_gt.getAnnIds(imgIds=image_id, iscrowd=None)
            matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == image_id]
            gt_ann_ids = [ann['id'] for ann in matching_anns]
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            # dt_ann_ids = coco_dt.getAnnIds(imgIds=image_id, iscrowd=None)
            matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == image_id]
            dt_ann_ids = [ann['id'] for ann in matching_anns]
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            # gt_cap_ann_ids = coco_cap_gt.getAnnIds(imgIds=image_id)
            matching_anns = [ann for ann in coco_cap_gt.anns.values() if ann['image_id'] == image_id]
            gt_cap_ann_ids = [ann['id'] for ann in matching_anns]
            gt_cap_ann = coco_cap_gt.loadAnns(gt_cap_ann_ids)[0]

            # dt_cap_ann_ids = coco_cap_dt.getAnnIds(imgIds=image_id)
            matching_anns = [ann for ann in coco_cap_dt.anns.values() if ann['image_id'] == image_id]
            dt_cap_ann_ids = [ann['id'] for ann in matching_anns]
            dt_cap_ann = coco_cap_dt.loadAnns(dt_cap_ann_ids)[0]

            gt_labels = gt_cap_ann['labels']
            dt_labels = dt_cap_ann['labels']

            actual_positives += len(gt_labels)

            # Find best matching pairs
            best_matches = find_best_matches(gt_anns, gt_labels, dt_anns, dt_labels, iou_threshold, text_sim_threshold)

            true_positives += len(best_matches)
        except Exception as e:
            print(e)

    recall = true_positives / actual_positives if actual_positives > 0 else 0

    print(f"Recall: {recall:.3f}")


def main():
    args = parse_args()

    # Get the image names of the split
    all_id_list = []
    all_images_list = []

    args.gt_json_path = os.path.join(args.root, args.gt_json_path)
    with open(args.gt_json_path, 'r') as f:
        contents = json.load(f)
        for image in contents['images']:
            all_id_list.append(image['id'])
            all_images_list.append(image['file_name'])

    print(len(all_id_list))
    # random.shuffle(all_images_list)

    # The directory is used to store intermediate files
    tmp_dir_path = f"tmp/{os.path.basename(args.prediction_dir_path)}_{args.split}"
    os.makedirs(tmp_dir_path, exist_ok=True)  # Create directory if not exists already

    # Create predictions
    pred_save_path = f"{tmp_dir_path}/mask_pred_tmp_save.json"
    cap_pred_save_path = f"{tmp_dir_path}/cap_pred_tmp_save.json"
    coco_pred_file = []
    caption_pred_dict = {}

    for idx, image_id in enumerate(all_id_list):
        # for json, we did not use id but us file name
        file_name = all_images_list[idx]
        prediction_path = f"{args.prediction_dir_path}/{file_name.split('/')[-1][:-4]}.json"

        if os.path.exists(prediction_path):
            print('prediction_path exist =============================>')

        try:
            with open(prediction_path, 'r') as f:
                pred = json.load(f)
                bu = pred
                key = list(pred.keys())[0]
                pred = pred[key]
                try:
                    caption_pred_dict[image_id] = {'caption': pred['caption'], 'labels': pred['phrases']}
                except Exception as e:
                    pred = bu
                    caption_pred_dict[image_id] = {'caption': pred['caption'], 'labels': pred['phrases']}
                for rle_mask in pred['pred_masks']:
                    coco_pred_file.append({"image_id": image_id, "category_id": 1, "segmentation": rle_mask, "score": 1.0})
        except:
            pass

    # Save gcg_coco_predictions
    with open(pred_save_path, 'w') as f:
        json.dump(coco_pred_file, f)

    # Prepare the CAPTION predictions in COCO format
    cap_image_ids = []
    coco_cap_pred_file = []
    for image_id, values in caption_pred_dict.items():
        cap_image_ids.append(image_id)
        coco_cap_pred_file.append({"image_id": image_id, "caption": values['caption'], "labels": values['labels']})

    # Save gcg_caption_coco_predictions
    with open(cap_pred_save_path, 'w') as f:
        json.dump(coco_cap_pred_file, f)

    # # -------------------------------#
    # 1. Evaluate AP
    # Calculate mask mAP
    # Load the ground truth and predictions in COCO format

    # # # -------------------------------#
    # # # Evaluate Caption Quality

    coco_cap_gt = COCO(args.gt_json_path)
    print(cap_pred_save_path)

    coco_cap_result = coco_cap_gt.loadRes(cap_pred_save_path)
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap_GRG(coco_cap_gt, coco_cap_result)
    coco_eval.params['image_id'] = coco_cap_result.getImgIds()
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')


    output_file_path = args.prediction_dir_path + '.txt'

    with open(output_file_path, 'w') as f:
        for metric, score in coco_eval.eval.items():
            f.write(f'{metric}: {score:.3f}\n')

    print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    main()
