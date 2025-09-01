#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python eval_unibiomed.py --num_samples -1 --dataset mecovqa_g
python eval_unibiomed.py --num_samples -1 --dataset VQA-RAD
python eval_unibiomed.py --num_samples -1 --dataset OmniMedVQA
python eval_unibiomed.py --num_samples -1 --dataset SLAKE
python eval_unibiomed.py --num_samples -1 --dataset imageclef
python eval_unibiomed.py --num_samples -1 --dataset PVQA