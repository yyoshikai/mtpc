import os
from typing import Optional
from collections.abc import Callable
from logging import getLogger
from argparse import ArgumentParser, Namespace
import torch
import torch.nn as nn
from torch import Tensor
from .backbone import Backbone
import torchvision.transforms as T
from ..data.image import RandomSolarization
from PIL.Image import Image

class BarlowTwinsTransform:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, resize_scale_min, resize_scale_max, resize_ratio_max, 
                example_dir=None, n_example: int=0):
        random_crop_size = (224, 224)
        color_plob, blur_plob, solar_plob = 0.8, 0.4, 0.0


        self.image_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=[0, 180]),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=color_plob),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur((3, 3), (1.0, 2.0))], p=blur_plob),
            RandomSolarization(prob=solar_plob),
            T.RandomResizedCrop(random_crop_size, scale=(resize_scale_min,
                    resize_scale_max), ratio=(1/resize_ratio_max, resize_ratio_max)),
        ])
        self.tensor_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.example_dir = example_dir
        self.n_example = n_example
        if self.n_example > 0:
            assert self.example_dir is not None
        self.n_saved = 0
    def __call__(self, x: Image):
        x_a = self.image_transform(x)
        x_b = self.image_transform(x)
        if self.n_saved < self.n_example:
            self.logger.info(f"saving {self.n_saved}..")
            os.makedirs(f"{self.example_dir}/{self.n_saved}", exist_ok=True)
            x.save(f"{self.example_dir}/{self.n_saved}/original.png")
            x_a.save(f"{self.example_dir}/{self.n_saved}/x0.png")
            x_b.save(f"{self.example_dir}/{self.n_saved}/x1.png")
            self.logger.info(f"saved.")
            self.n_saved += 1
        x_a = self.tensor_transform(x_a)
        x_b = self.tensor_transform(x_b)
        return x_a, x_b

class BarlowTwins(nn.Module):
    def __init__(self, backbone: Backbone, lambda_param: float, 
            head_size: int, 
            resize_scale_min, resize_scale_max, resize_ratio_max):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(backbone.output_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, head_size)
        )
        self.lambda_param = lambda_param
        self.resize_scale_min = resize_scale_min
        self.resize_scale_max = resize_scale_max
        self.resize_ratio_max = resize_ratio_max

    def forward(self, x: tuple[Tensor, Tensor]):
        x_a, x_b = x
        device = self.head[0].weight.device
        x_a = x_a.to(device)
        x_b = x_b.to(device)

        # model
        z_a: Tensor = self.head(self.backbone(x_a))
        z_b: Tensor = self.head(self.backbone(x_b))

        # criterion
        device = z_a.device

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD
        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        
        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss
    
    def get_train_transform(self, example_dir: Optional[str], n_example: int) -> Callable[[Image], tuple[Tensor, Tensor]]:
        return BarlowTwinsTransform(self.resize_scale_min, self.resize_scale_max, self.resize_ratio_max, example_dir, n_example)
    
    def get_eval_transform(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument('--head-size', type=int, default=128)
        parser.add_argument('--lambda-param', type=float, default=5e-3)
        ## augmentation
        parser.add_argument('--resize-scale-min', type=float, default=0.08)
        parser.add_argument('--resize-scale-max', type=float, default=1.0)
        parser.add_argument('--resize-ratio-max', type=float, default=4/3)

    @classmethod
    def from_args(cls, backbone: Backbone, args: Namespace=None):
        if args is None:
            parser = ArgumentParser()
            cls.add_args(parser)
            args = parser.parse_args([])
        return cls(backbone, args.lambda_param, args.head_size, 
                args.resize_scale_min, args.resize_scale_max, args.resize_ratio_max)


