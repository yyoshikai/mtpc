# -*- coding: utf-8 -*-
"""
# barlowtwins with recursive resuming

@author: Katsuhisa MORITA
"""

# packages installed in the current environment
import sys
import os
import gc
import argparse
import random
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# path setting
WORKDIR = os.environ.get('WORKDIR', "/workspace")
PROJECT_PATH = f'{WORKDIR}/mtpc/pretrain/SelfSupervisedLearning_Pathology'

# original packages in src
sys.path += [PROJECT_PATH, f"{WORKDIR}/mtpc"]
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
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class RandomSolarization(object):
    def __init__(self, prob: float = 0.5, threshold: int = 128):
        self.prob = prob
        self.threshold = threshold

    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < self.prob:
            return ImageOps.solarize(sample, threshold=self.threshold)
        return sample

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def ssl_transform(
    size=(224,224),
    color_plob=0.8,
    blur_plob=0.2,
    solar_plob=0,
    ):
    # augmentation
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=[0, 180])], p=1.),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=color_plob
            ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=blur_plob
            ),
        RandomSolarization(prob=solar_plob),
        transforms.RandomResizedCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    ])
    return TwoCropsTransform(augmentation)


class BarlowTwins(nn.Module):
    def __init__(self, head_size=[2048, 512, 128]):
        super().__init__()
        encoder = torchvision.models.resnet50()
        self.backbone = nn.Sequential(*list(encoder.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(head_size[0], head_size[1], bias=False), 
            nn.BatchNorm1d(head_size[1]), 
            nn.ReLU(),
            nn.Linear(head_size[1], head_size[1], bias=False), 
            nn.BatchNorm1d(head_size[1]), 
            nn.ReLU(),
            nn.Linear(head_size[1], head_size[2], bias=True), 
        )
        self.lambda_param = 5e-3

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1).flatten(start_dim=1))
        z2 = self.projector(self.backbone(x2).flatten(start_dim=1))
        
        device = z1.device
        z_a_norm = (z1 - z1.mean(0)) / z1.std(0) # NxD
        z_b_norm = (z2 - z2.mean(0)) / z2.std(0) # NxD
        N = z1.size(0)
        D = z1.size(1)

        c = z_a_norm.T @ z_b_norm # DxD
        c.div_(N)
        c_diff = (c - torch.eye(D, device=device)).pow(2) # DxD
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
        return loss

if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device

    # 1. Preparing
    model = BarlowTwins(head_size=[2048, 512, 128])
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=0)

    # 2. Training
    for epoch in range(0, args.num_epoch):
        # train
        model.train() # training

        random.seed(args.seed)
        # normalization
        train_transform = ssl_transform(
            color_plob=args.color_plob,
            blur_plob=args.blur_plob, 
            solar_plob=args.solar_plob
        )
        # data
        train_dataset = TGGATEDataset(args.data)
        train_dataset = ApplyDataset(train_dataset, train_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            sampler=None,
        )

        # train
        for i, data in enumerate(train_loader):
            x1, x2 = data[0].to(DEVICE), data[1].to(DEVICE) # put data on GPU
            loss = model(x1, x2)

            print(f"{i=}, {loss.item()=}", flush=True)
            optimizer.zero_grad() # reset gradients
            loss.backward() # backpropagation
            optimizer.step() # update parameters
        del train_loader
        gc.collect()
        scheduler.step()
