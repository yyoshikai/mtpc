import os
from typing import Any, TypeVar
from collections.abc import Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageOps
from .data import WrapDataset

T_co = TypeVar('T_co', covariant=True)

class RandomSolarization(object):
    def __init__(self, prob: float = 0.5, threshold: int = 128):
        self.prob = prob
        self.threshold = threshold

    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < self.prob:
            return ImageOps.solarize(sample, threshold=self.threshold)
        else:
            return sample

class BaseAugmentDataset(Dataset[torch.Tensor]):
    def __init__(self, dataset: Dataset[Image.Image]):
        self.dataset = dataset
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx: int):
        return self.transform(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

class TransformDataset(WrapDataset[T_co]):
    def __init__(self, dataset: Dataset[Image.Image], transform: Callable[[Image.Image], T_co]):
        super().__init__(dataset)
        self.transform = transform

    def __getitem__(self, idx: int):
        return self.transform(self.dataset[idx])

