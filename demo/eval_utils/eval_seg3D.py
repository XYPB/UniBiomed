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
    for volume_name in save_metrics.keys():
        each_volume = save_metrics[volume_name]
        for class_name in each_volume.keys():
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

        for volume_name in save_metrics.keys():
            each_volume = save_metrics[volume_name]
            for class_name in each_volume.keys():
                if class_name == per_class_name:
                    dice_per_class_list.append(each_volume[class_name]['dice'])
                    iou_per_class_list.append(each_volume[class_name]['IoU'])

                all_dice.append(each_volume[class_name]['dice']), all_iou.append(each_volume[class_name]['IoU'])

        dice_per_class = sum(dice_per_class_list) / (len(dice_per_class_list)+1e-6)
        iou_per_class = sum(iou_per_class_list) / (len(iou_per_class_list) + 1e-6)

        print(per_class_name, dice_per_class)
        per_class_metrics['dice'] = dice_per_class
        per_class_metrics['IoU'] = iou_per_class

        all_class_dice.append(dice_per_class)
        all_class_iou.append(iou_per_class)

        mean_metrics[per_class_name] = per_class_metrics

    mean_class_dice = sum(all_class_dice) / (len(all_class_dice) + 1e-6)
    mean_class_iou = sum(all_class_iou) / (len(all_class_iou) + 1e-6)
    mean_metrics['mean_class_dice'] = mean_class_dice
    mean_metrics['mean_class_iou'] = mean_class_iou

    mean_dice = sum(all_dice) / (len(all_dice) + 1e-6)
    mean_iou = sum(all_iou) / (len(all_iou) + 1e-6)
    mean_metrics['mean_dice'] = mean_dice
    mean_metrics['mean_iou'] = mean_iou

    print('mean_class_dice, mean_class_iou:', mean_class_dice, mean_class_iou)
    print('mean_dice, mean_class_iou:', mean_dice, mean_iou)

    return mean_metrics


def save_metrics_json(data_root, results_root, val_json_file='meta_train.json'):
    save_json_file = os.path.join(results_root, 'results.json')

    json_file = os.path.join(data_root, val_json_file)
    with open(json_file, 'r') as file:
        data = json.load(file)

    volumes_names = data['volumes'].keys()

    save_metrics = {}

    for volume_name in volumes_names:
        volume = data['volumes'][volume_name]
        expressions = volume['expressions']

        volume_metrics = {}

        for exp_id in expressions.keys():
            exp = expressions[exp_id]

            exp_sentence = exp['exp']
            mask_list = exp['mask_file']
            img_list = exp['img_file']
            assert len(mask_list) == len(img_list), print(len(mask_list), len(img_list))

            volume_masks_per_class = []
            volume_preds_per_class = []

            for i in range(len(mask_list)):
                mask = os.path.join(data_root, mask_list[i])
                pred = os.path.join(results_root, volume_name, exp_sentence, img_list[i].split('/')[-1])

                mask, pred = Image.open(mask), Image.open(pred)
                mask, pred = np.asarray(mask), np.asarray(pred)
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]

                if mask.shape != pred.shape:
                    mask = cv2.resize(mask, (pred.shape[1], pred.shape[0]))

                mask, pred = (mask > 0).astype(np.uint8), (pred > 0).astype(np.uint8)

                volume_masks_per_class.append(mask), volume_preds_per_class.append(pred)

            volume_masks_per_class = np.stack(volume_masks_per_class, axis=0)
            volume_preds_per_class = np.stack(volume_preds_per_class, axis=0)

            if volume_masks_per_class.sum() > 0:
                dice, IoU = get_metrics(volume_preds_per_class, volume_masks_per_class)
                print('volume_name, exp, dice, IoU:', volume_name, exp_sentence, dice, IoU)
                exp_metric = {}
                exp_metric['dice'] = dice
                exp_metric['IoU'] = IoU

                volume_metrics[exp_sentence] = exp_metric

        save_metrics[volume_name] = volume_metrics
        print(volume_name, volume_metrics)
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
    data_root = '/data/linshan/Biomedparse/3D/CHAOS'
    results_root = '/home/linshan/Sa2VA/val_results/3D/' + data_root.split('/')[-1]
    save_metrics_json(data_root, results_root)

