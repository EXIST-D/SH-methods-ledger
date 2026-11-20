import argparse, random
import numpy as np
import torch
from torch.optim import AdamW

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml', help='config path')
    p.add_argument('--epochs', type=int, default=40, help='number of epochs')
    p.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--amp', action='store_true')
    return p.parse_args()

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(0)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
    sched = CosineAnnealingLR(opt, T_max=50)
    net = nn.Dropout(0.5)
