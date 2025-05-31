# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from argparse import ArgumentParser, Namespace
from typing import Optional
from logging import getLogger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .backbone import Backbone
from PIL import ImageFilter
from PIL.Image import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from ..data.image import RandomSolarization

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class VICRegTransform(object):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, example_dir: Optional[str], n_example: int):
        self.transform = T.Compose([
            T.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8,),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            RandomSolarization(prob=0.0),
        ])
        self.transform_prime = T.Compose([
            T.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8,),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            RandomSolarization(prob=0.2),
        ])
        self.tensor_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.example_dir = example_dir
        self.n_example = n_example
        if self.example_dir is None:
            assert self.n_example == 0
        self.n_saved = 0

    def __call__(self, x: Image):
        x1 = self.transform(x)
        x2 = self.transform_prime(x)
        if self.n_saved < self.n_example:
            os.makedirs(f"{self.example_dir}/{self.n_saved}", exist_ok=True)
            x.save(f"{self.example_dir}/{self.n_saved}/original.png")
            x1.save(f"{self.example_dir}/{self.n_saved}/x0.png")
            x2.save(f"{self.example_dir}/{self.n_saved}/x1.png")
            self.n_saved += 1
        x1 = self.tensor_transform(x1)
        x2 = self.tensor_transform(x2)
        return x1, x2

class VICReg(nn.Module):
    """
    from https://github.com/facebookresearch/vicreg
    """
    def __init__(self, backbone: Backbone, head_sizes: list[int], 
                sim_coeff, std_coeff, cov_coeff):
        super().__init__()
        self.backbone = backbone
        head_sizes = [backbone.output_size]+head_sizes
        head_layers = []
        for i in range(len(head_sizes)-2):
            head_layers += [
                nn.Linear(head_sizes[i], head_sizes[i+1]),
                nn.BatchNorm1d(head_sizes[i+1]),
                nn.ReLU(True)
            ]
        head_layers.append(nn.Linear(head_sizes[-2], head_sizes[-1], bias=False))
        self.head = nn.Sequential(*head_layers)
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x: tuple[Tensor, Tensor]):
        x, y = x
        device = self.head[0].weight.device
        x = x.to(device)
        y = y.to(device)
        
        x: Tensor = self.head(self.backbone(x))
        y: Tensor = self.head(self.backbone(y))
        B, D = x.shape

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (B - 1)
        cov_y = (y.T @ y) / (B - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(D) \
                + off_diagonal(cov_y).pow_(2).sum().div(D)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss
    
    def get_train_transform(self, example_dir, n_example):
        return VICRegTransform(example_dir, n_example)
    
    def get_eval_transform(self):
        raise NotImplementedError

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument('--head-sizes', type=int, nargs='+', 
                default=[8192, 8192, 8192])
        parser.add_argument('--sim-coeff', type=float, default=25.0)
        parser.add_argument('--std-coeff', type=float, default=25.0)
        parser.add_argument('--cov-coeff', type=float, default=1.0)

    @classmethod
    def from_args(cls, args: Namespace, backbone: Backbone):
        return VICReg(backbone, args.head_sizes, args.sim_coeff, 
                    args.std_coeff, args.cov_coeff)



def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


