## Experimental setup

All models were trained for 100 epochs using the Adam optimizer with a learning rate of 0.01 and a batch size of 64. The random seed was 7. Experiments were run on 8 NVIDIA V100 GPUs with PyTorch 2.0 and Python 3.8.

## Dataset

We use LS-Tiles v2.0, which contains 12,000 tiles. The data were split 7:2:1 into training, validation and test sets. During evaluation we use a batch size of 32. Each image is resized to 224 pixels.

## Results and comparison

LiteSlide reaches an F1 of 0.85 and an accuracy of 87.1%, against 0.771 for BaselineA under the same budget. XX ablations are in the supplement.

## Limitations

The method may not generalize to other regions. Further research is needed.
