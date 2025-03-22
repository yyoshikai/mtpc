import sys, os, argparse, yaml
import torch.nn as nn

from torch.utils.data import Dataset, ConcatDataset
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/mtpc"]
from src.data.mtpc import MTPCUHRegionDataset, MTPCVDRegionDataset

datas = []
for wsi_idx in range(1, 106):
    datas.append([MTPCUHRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)])
for wsi_idx in range(1, 55):
    datas.append([MTPCVDRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)])
data = ConcatDataset(datas)
data = BaseAugmentDataset(data)


# Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(2048, 128),
            nn.GELU(),
            nn.Linear(128, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x.squeeze_(-1, -2)
        x = self.head(x)
        x.squeeze_(-1)
        return x
