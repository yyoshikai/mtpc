import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageOps


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


class AugmentDataset(Dataset[torch.Tensor]):
    def __init__(self, dataset: Dataset[Image.Image], 
        color_plob, blur_plob, solar_plob, random_crop_size,
        resize_scale_min, resize_scale_max, resize_ratio_max, 
        example_dir: str=None, n_example: int=3):
        self.dataset = dataset
        self.image_aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=[0, 180]),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=color_plob
                ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([
                T.GaussianBlur((3, 3), (1.0, 2.0))], p=blur_plob
                ),
            RandomSolarization(prob=solar_plob),
            T.RandomResizedCrop(random_crop_size, scale=(resize_scale_min, resize_scale_max), ratio=(1/resize_ratio_max, resize_ratio_max)),
        ])
        self.tensor_aug = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ])

        self.example_dir = example_dir
        if self.example_dir is not None:
            self.n_example = n_example
        else:
            self.n_example = 0
        self.n_saved = 0

    def __getitem__(self, idx):
        image = self.dataset[idx]
        aug_image = self.image_aug(image)
        if self.n_saved < self.n_example:
            os.makedirs(f"{self.example_dir}/{idx}", exist_ok=True)
            image.save(f"{self.example_dir}/{idx}/original.png")
            aug_image.save(f"{self.example_dir}/{idx}/aug.png")
            self.n_saved += 1
        return self.tensor_aug(aug_image)
    
    def __len__(self):
        return len(self.dataset)

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
