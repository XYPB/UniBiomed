from tqdm import tqdm
import os
import numpy as np
import json
from PIL import Image
import cv2


def get_metrics(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()

    dice = 2.0 * intersection / (prediction.sum() + target.sum())
    IoU = intersection / union

    return dice, IoU


### GET MEAN DICE AND IOU !
def get_mean(save_metrics):
    mean_metrics = {}
    classes_list = []

    # first get class names
    for name in save_metrics.keys():
        each_item = save_metrics[name]
        if 'exp' in each_item.keys():
            class_name = each_item['exp']
            if class_name not in classes_list:
                classes_list.append(class_name)

    all_class_dice = []
    all_class_iou = []

    all_dice = []
    all_iou = []

    for per_class_name in classes_list:
        per_class_metrics = {}

        dice_per_class_list = []
        iou_per_class_list = []

        for name in save_metrics.keys():
            each_item = save_metrics[name]
            if 'exp' in each_item.keys():
                class_name = each_item['exp']

                if class_name == per_class_name:
                    dice_per_class_list.append(each_item['dice'])
                    iou_per_class_list.append(each_item['IoU'])

                all_dice.append(each_item['dice']), all_iou.append(each_item['IoU'])

        dice_per_class = sum(dice_per_class_list) / (len(dice_per_class_list)+1e-6)
        iou_per_class = sum(iou_per_class_list) / (len(iou_per_class_list) + 1e-6)

        print(per_class_name, dice_per_class)
        per_class_metrics['dice'] = dice_per_class
        per_class_metrics['IoU'] = iou_per_class

        all_class_dice.append(dice_per_class)
        all_class_iou.append(iou_per_class)

        mean_metrics[per_class_name] = per_class_metrics

    mean_class_dice = sum(all_class_dice) / (len(all_class_dice)+1e-6)
    mean_class_iou = sum(all_class_iou) / (len(all_class_iou) + 1e-6)
    mean_metrics['mean_class_dice'] = mean_class_dice
    mean_metrics['mean_class_iou'] = mean_class_iou

    mean_dice = sum(all_dice) / (len(all_dice) + 1e-6)
    mean_iou = sum(all_iou) / (len(all_iou) + 1e-6)
    mean_metrics['mean_dice'] = mean_dice
    mean_metrics['mean_iou'] = mean_iou

    print('mean_class_dice, mean_class_iou:', mean_class_dice, mean_class_iou)
    print('mean_dice, mean_iou:', mean_dice, mean_iou)

    return mean_metrics


def save_metrics_json(data_root, results_root, val_json_file='train.json'):
    save_json_file = os.path.join(results_root, 'results.json')

    json_file = os.path.join(data_root, val_json_file)
    with open(json_file, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']

    save_metrics = {}

    for item in annotations[:300]:
        item_metric = {}

        mask_name = item['mask_file']

        mask = os.path.join(data_root, item['split']+'_mask', mask_name)
        pred = os.path.join(results_root, item['split'], mask_name)

        mask, pred = Image.open(mask), Image.open(pred)
        mask, pred = np.asarray(mask), np.asarray(pred)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if mask.shape != pred.shape:
            mask = cv2.resize(mask, (pred.shape[1], pred.shape[0]))

        mask, pred = (mask > 0).astype(np.uint8), (pred > 0).astype(np.uint8)

        if mask.sum() > 0:
            dice, IoU = get_metrics(pred, mask)
            print('mask_name, dice, IoU:', mask_name, dice, IoU)

            item_metric['dice'] = dice
            item_metric['IoU'] = IoU
            item_metric['exp'] = item['sentences'][0]['sent']

        save_metrics[mask_name] = item_metric
        print(item_metric)
        print('\n')

    mean_metrics = get_mean(save_metrics)

    # new: want mean metrics in the first line
    new_save_metrics = {}
    new_save_metrics['mean'] = mean_metrics

    # new: want mean metrics in the first line
    for key in save_metrics.keys():
        new_save_metrics[key] = save_metrics[key]

    with open(save_json_file, 'w', encoding='utf-8') as json_file:
        json.dump(new_save_metrics, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    data_root = '/data/linshan/Biomedparse/CHAOS'
    results_root = '/home/linshan/Sa2VA/val_results/2D/'+data_root.split('/')[-1]
    save_metrics_json(data_root, results_root)

