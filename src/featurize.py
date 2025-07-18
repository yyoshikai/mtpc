import sys, os
from logging import getLogger
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm as _tqdm
from torch.utils.data import DataLoader, ConcatDataset
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/mtpc"]
from src.data.mtpc import MTPCDataset, MTPCUHRegionDataset, MTPCVDRegionDataset
from src.data.image import TransformDataset
from src.data import untuple_dataset
from src.utils.time import wtqdm
CDIR = str(Path(__file__).parents[1] / 'featurize')

base_main_data = MTPCDataset(256)
base_main_data = untuple_dataset(base_main_data, 2)[0]

add_datas = []
for wsi_idx in range(1, 106):
    for region_idx in range(1, 4):
        data = MTPCUHRegionDataset(wsi_idx, region_idx)
        add_datas.append(data)
for wsi_idx in range(1, 55):
    for region_idx in range(1, 4):
        data = MTPCVDRegionDataset(wsi_idx, region_idx)
        add_datas.append(data)
base_add_data = ConcatDataset(add_datas)

def featurize_mtpc(fname, num_workers, batch_size, backbone, transform, tqdm=True):

    out_dir = f"{CDIR}/{fname}"
    logger = getLogger('featurize_mtpc')
    
    # check result exists
    if os.path.exists(f"{out_dir}/feat_all.npy") and os.path.exists(f"{out_dir}/feat_added.npy"):
        logger.info(f"Already featurized: {out_dir}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"{device=}")

    if batch_size is None:
        batch_size = 128
    
        
    # model
    backbone.to(device)
    backbone.eval()

    # main data
    main_data = TransformDataset(base_main_data, transform)
    loader = DataLoader(main_data, shuffle=False, batch_size=batch_size, 
            num_workers=num_workers, prefetch_factor=None if num_workers==0 else 10)
    feats = []
    with torch.inference_mode():
        for batch in (pbar:=(wtqdm(loader) if tqdm else loader)):
            if tqdm: pbar.start('featurize')
            feats.append(backbone(batch.to(device)).cpu().numpy())

    feat = np.concatenate(feats, axis=0)
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/feat_all.npy", feat)
    
    # sub data
    add_data = TransformDataset(base_add_data, transform)
    loader = DataLoader(add_data, shuffle=False, batch_size=batch_size, 
            num_workers=num_workers, prefetch_factor=None if num_workers==0 else 10)

    feats = []
    with torch.no_grad():
        for batch in (pbar:=(wtqdm(loader) if tqdm else loader)):
            if tqdm: pbar.start('featurize')
            feat = backbone(batch.to(device)).cpu().numpy()
            feats.append(feat)
    feat = np.concatenate(feats, axis=0)

    np.save(f"{out_dir}/feat_added.npy", feat)