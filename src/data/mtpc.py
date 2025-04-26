import sys, os
from logging import getLogger
from typing import TypeVar
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np, pandas as pd
from PIL import Image
from tifffile import TiffFile
from ..utils.logger import get_logger

WORKDIR = os.environ.get('WORKDIR', "/workspace")
DDIR = f"{WORKDIR}/cheminfodata/mtpc"

T_co = TypeVar('T_co', covariant=True)

class MTPCRegionDataset(Dataset[Image.Image]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, case_name: str, region: int, psz: int):
        self.patch_poss = None
        self.wsi_arr = None
        self.psz = psz
        self.case_name = case_name
        self.region = region
    
    def __getitem__(self, idx:int) -> Image.Image:
        if self.patch_poss is None:
            self.patch_poss = np.load(f"{DDIR}/processed/patch/color_s0/{self.case_name}/{self.region}.npy")
        if self.wsi_arr is None:
            with TiffFile(f"{DDIR}/raw/Project_Output/{self.case_name}/{self.case_name}_Region_{self.region}.tiff") as tif:
                self.wsi_arr = tif.asarray(out='memmap')

        y, x = self.patch_poss[idx].tolist()
        patch_arr = np.array(self.wsi_arr[y:y+self.psz, x:x+self.psz])
        return Image.fromarray(patch_arr)
    
    def __len__(self) -> int:
        if self.patch_poss is None:
            self.patch_poss = np.load(f"{DDIR}/processed/patch/color_s0/{self.case_name}/{self.region}.npy")
        return len(self.patch_poss)  


class MTPCUHRegionDataset(Dataset[Image.Image]):
    def __init__(self, wsi_idx, region_idx):
        self.wsi_idx = wsi_idx
        self.region_idx = region_idx
        self.image = None
        self.patch_poss = np.load(f"{DDIR}/UH061/HE_sat_patch_idx/{self.wsi_idx}/{self.region_idx}.npy")
        self.psz = 256
    
    def __getitem__(self, idx):
        if self.image is None:
            self.image = Image.open(f"{DDIR}/UH061/HE/{self.wsi_idx}/{self.region_idx}.jpg")
        pos = self.patch_poss[idx]
        y, x = pos[0], pos[1]
        return self.image.crop((x, y, x+self.psz, y+self.psz), )
    
    def __len__(self):
        return len(self.patch_poss)

class MTPCVDRegionDataset(Dataset[Image.Image]):
    def __init__(self, wsi_idx, region_idx):
        with TiffFile(f"{DDIR}/VD261/HE/{wsi_idx}/{region_idx}.tif") as tif:
            self.wsi_arr = tif.asarray(out='memmap')
        self.patch_poss = np.load(f"{DDIR}/VD261/HE_sat_patch_idx/{wsi_idx}/{region_idx}.npy")
        self.psz = 256
    
    def __getitem__(self, idx):
        y, x = self.patch_poss[idx, 0], self.patch_poss[idx, 1]
        return Image.fromarray(self.wsi_arr[y:y+self.psz, x:x+self.psz])
    def __len__(self):
        return len(self.patch_poss)

# 今はtrain.pyでは使っていない。
class MTPCCaseDataset(ConcatDataset):
    def __init__(self, case_name: str, psz: int, label: str):
        df = pd.read_csv(f"{DDIR}/processed/annotation_check0.csv", index_col=0, 
                dtype=str, keep_default_na=False)
        regions = df.columns.values[df.loc[case_name] != 'NaN']
        self.label = label
        super().__init__([MTPCRegionDataset(case_name, region, psz) for region in regions])
    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        return item, self.label

# 今はtrain.pyでは使っていない。
class MTPCDataset(ConcatDataset):
    def __init__(self, psz: int):
        df = pd.read_csv(f"{DDIR}/processed/annotation_check_meta0.csv", index_col=0, dtype=str)
        super().__init__([MTPCCaseDataset(case_name, psz, label) for case_name, label in zip(df.index, df['Pathological class'])])



class InDataset(Dataset[int]):
    logger = get_logger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset: Dataset, positive_values: list):
        self.dataset = dataset
        self.positive_values = positive_values
        self.logger.info(f"{positive_values=}")

    def __getitem__(self, idx: int):
        return int(self.dataset[idx] in self.positive_values)

    def __len__(self):
        return len(self.dataset)

class OrdinalDataset(Dataset[torch.Tensor]):
    def __init__(self, dataset: Dataset, categories: list):
        self.dataset = dataset
        self.categories = categories
        self.size = len(self.categories)-1
    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        i = self.categories.index(item)
        item = np.zeros(self.size, dtype=int)
        item[:i] = 1
        return item
    
    def __len__(self):
        return len(self.dataset)

class ShuffleAugmentDataset(Dataset[tuple[T_co, T_co]]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset: Dataset[T_co]):
        self.dataset = dataset
        self.aug_idx_iter = iter([])
        self.aug_epoch = -1

    def augment_idx(self):
        try:
            return self.aug_idx_iter.__next__()
        except StopIteration:
            self.aug_epoch += 1
            aug_idxs = list(range(len(self.dataset)))
            rng = np.random.default_rng(self.aug_epoch)
            rng.shuffle(aug_idxs)
            self.aug_idx_iter = aug_idxs.__iter__()
            return self.aug_idx_iter.__next__()

    def __getitem__(self, idx: int) -> tuple[T_co, T_co]:
        return self.dataset[idx], self.dataset[self.augment_idx()]

    def __len__(self):
        return len(self.dataset)

class FixedShuffleAugmentDataset(ShuffleAugmentDataset):
    def __init__(self, dataset: Dataset[T_co], epoch: int=-1):
        super().__init__(dataset)
        self.epoch = epoch
    def __getitem__(self, idx: int):
        super().__getitem__((idx, self.epoch))

