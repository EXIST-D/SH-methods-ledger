## Experimental setup

LiteSlide was trained for 50 epochs with AdamW (learning rate 1e-3, weight decay 0.01 [[CONFIRM: weight decay - 0.01 in config.yaml, 0.05 hard-coded in the AdamW call of train.py]], cosine schedule with 500 warm-up steps) and a batch size of 32 [[CONFIRM: training batch size - config.yaml says 32, the DataLoader call in train.py says 16]]. Evaluation used a batch size of 64. The random seed was 42. Experiments ran on 4 NVIDIA A100-SXM4-40GB GPUs with PyTorch 2.1.0, CUDA 12.1 and Python 3.10.12. The backbone is resnet18 with dropout [[CONFIRM: dropout - 0.2 in config.yaml, 0.5 in train.py]]. Each image is resized to 256 pixels.

## Dataset

We use LS-Tiles v2.1, which contains 10,000 labelled tiles from 14 regions, split 8:1:1 at random into 8,000 training, 1,000 validation and 1,000 test tiles [[CONFIRM: how the split was drawn]].

## Baselines and evaluation

On the test set LiteSlide reaches an F1 of 0.812 with 4.1 million parameters, against 0.771 for BaselineA (11.2 million parameters). Test accuracy is 0.871. [[CONFIRM: baseline budget and training settings]] [[CONFIRM: number of repeated runs and statistics (mean and std)]]

## Reproducibility

Run `python train.py --config config.yaml --epochs 50 --lr 0.001 --seed 42`, then `python eval.py --ckpt best.pt --batch-size 64`. [[CONFIRM: code version (commit)]] [[CONFIRM: how to obtain LS-Tiles]]

## Limitations

The model was tested only on Sentinel imagery; when the target region lies outside the 14 training regions, F1 can drop (not measured here).
