import sys, os, argparse, yaml

from torch.utils.data import Dataset, ConcatDataset
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/mtpc"]
from src.data.mtpc import MTPCUHRegionDataset, MTPCVDRegionDataset

datas = []
for wsi_idx in range(1, 106):
    datas.append([MTPCUHRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)])
for wsi_idx in range(1, 55):
    datas.append([MTPCVDRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)])
image_data = ConcatDataset(datas)


