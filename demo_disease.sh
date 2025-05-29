#!/bin/bash

DATA_LIST=("ColonCancer"
"LiverTumor" "LungTumor"
"pneumothorax" "ProstateCancer" "Retinal" "BreastTumor"
"COVID19"
"KidneyTumor" "LungNodule" "PancreasTumor"
"ColonPolyp" "Skin" "Fibrotic" "BrainTumor" 'NoFindings')

for name in "${DATA_LIST[@]}"; do
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 python demo/demo_disease.py --data_path ./data/Biomed/Disease/$name --model_path ./save_hf --save_dir ./val_results/Grounded_disease/$name
done

for name in "${DATA_LIST[@]}"; do
    PYTHONPATH=. python demo/eval_utils/metrics_grounded_disease.py  --root ./data/Biomed/Disease/$name --prediction_dir_path ./val_results/Grounded_disease/$name
done