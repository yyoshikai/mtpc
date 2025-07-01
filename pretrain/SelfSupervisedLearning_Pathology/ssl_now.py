# -*- coding: utf-8 -*-
"""
# barlowtwins with recursive resuming

@author: Katsuhisa MORITA
"""

# packages installed in the current environment
import sys
import os
import gc
import datetime
import argparse
import time
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image

# path setting
WORKDIR = os.environ.get('WORKDIR', "/workspace")
PROJECT_PATH = f'{WORKDIR}/mtpc/pretrain/SelfSupervisedLearning_Pathology'

# original packages in src
sys.path += [PROJECT_PATH, f"{WORKDIR}/mtpc"]
import sslmodel
from sslmodel import data_handler as dh
import sslmodel.sslutils as sslutils
from sslmodel.models import barlowtwins
from src.data.tggate import TGGATEDataset
from src.data import ApplyDataset

# argument
parser = argparse.ArgumentParser(description='CLI learning')
# base settings
parser.add_argument('--seed', type=int, default=0)
# model/learning settings
parser.add_argument('--num_epoch', type=int, default=50) # epoch
parser.add_argument('--batch_size', type=int, default=64) # batch size
parser.add_argument('--lr', type=float, default=0.01) # learning rate
# Transform (augmentation) settings
parser.add_argument('--color_plob', type=float, default=0.8)
parser.add_argument('--blur_plob', type=float, default=0.4)
parser.add_argument('--solar_plob', type=float, default=0.)
parser.add_argument('--data', default="/workspace/patho/preprocess/results"
        "/tggate_liver_late")

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device

    # 1. Preparing
    encoder = torchvision.models.resnet50(weights=None)
    backbone = nn.Sequential(*list(encoder.children())[:-1])
    model = barlowtwins.BarlowTwins(backbone, head_size=[2048, 512, 128])
    criterion = barlowtwins.BarlowTwinsLoss()
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=0)

    # 2. Training
    for epoch in range(0, args.num_epoch):
        # train
        model.train() # training

        random.seed(args.seed)
        # normalization
        train_transform = sslmodel.utils.ssl_transform(
            color_plob=args.color_plob,
            blur_plob=args.blur_plob, 
            solar_plob=args.solar_plob,
            split=True, multi=False,
        )
        # data
        train_dataset = TGGATEDataset(args.data)
        if not isinstance(train_transform, list): 
            train_transform = [train_transform]
        for t in train_transform:
            train_dataset = ApplyDataset(train_dataset, t)

        train_loader = dh.prep_dataloader(train_dataset, args.batch_size)

        # train
        for i, data in enumerate(train_loader):
            x1, x2 = data[0].to(DEVICE), data[1].to(DEVICE) # put data on GPU
            z1, z2 = model(x1), model(x2)
            loss = criterion(z1, z2)

            print(f"{i=}, {loss.item()=}", flush=True)
            optimizer.zero_grad() # reset gradients
            loss.backward() # backpropagation
            optimizer.step() # update parameters
        del train_loader
        gc.collect()
        scheduler.step()
