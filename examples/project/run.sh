#!/bin/bash
# reproduce Table 1
python train.py --config config.yaml --epochs 50 --lr 0.001 --seed 42
python eval.py --ckpt best.pt --batch-size 64
