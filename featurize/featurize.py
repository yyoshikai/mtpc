import sys, os, yaml
from argparse import ArgumentParser
from logging import getLogger
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from addict import Dict
from tqdm import tqdm
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/mtpc"]
from src.data.mtpc import MTPCDataset, MTPCUHRegionDataset, MTPCVDRegionDataset
from src.data.image import TransformDataset
from src.data import untuple_dataset
from src.model.backbone import get_backbone
from src.utils.model import get_substate
from src.model.barlowtwins import BarlowTwins
from src.model.vicreg import VICReg
from src.model.vicregl import VICRegL
scheme_name2cls = {
    'bt': BarlowTwins,
    'vicreg': VICReg, 
    'vicregl': VICRegL
}

CDIR = os.path.dirname(__file__)

def featurize_mtpc(fname, num_workers, batch_size, backbone, transform):

    logger = getLogger('featurize_mtpc')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"{device=}")

    if batch_size is None:
        batch_size = 128
    
    out_dir = f"{CDIR}/{fname}"
        
    # model
    backbone.to(device)
    backbone.eval()

    # main data
    dataset = MTPCDataset(256)
    dataset = untuple_dataset(dataset, 2)[0]
    dataset = TransformDataset(dataset, transform)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, 
            num_workers=num_workers, prefetch_factor=None if num_workers==0 else 10)
    feats = []
    with torch.inference_mode():
        for batch in tqdm(loader):
            feats.append(backbone(batch.to(device)).cpu().numpy())

    feat = np.concatenate(feats, axis=0)
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/feat_all.npy", feat)
    
    # sub data
    datas = []
    for wsi_idx in range(1, 106):
        for region_idx in range(1, 4):
            data = MTPCUHRegionDataset(wsi_idx, region_idx)
            datas.append(data)
    for wsi_idx in range(1, 55):
        for region_idx in range(1, 4):
            data = MTPCVDRegionDataset(wsi_idx, region_idx)
            datas.append(data)
    dataset = ConcatDataset(datas)

    dataset = TransformDataset(dataset, transform)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, 
            num_workers=num_workers, prefetch_factor=None if num_workers==0 else 10)

    feats = []
    with torch.no_grad():
        for batch in tqdm(loader):
            feat = backbone(batch.to(device)).cpu().numpy()
            feats.append(feat)
    feat = np.concatenate(feats, axis=0)

    np.save(f"{out_dir}/feat_added.npy", feat)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--sname', required=True)
    parser.add_argument('--epoch', type=int, default=29)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--remove-last-relu', action='store_true')
    parser.add_argument('--from-imagenet')
    parser.add_argument('--bsz', type=int)
    args = parser.parse_args()

    rdir = f"/workspace/mtpc/pretrain/bt/results/{args.sname}"
    
    with open(f"{rdir}/args.yaml") as f:
        pargs = Dict(yaml.safe_load(f))
    batch_size = args.bsz or pargs['bsz']

    snameh = args.sname
    if args.remove_last_relu: snameh += '_norelu'

    structure = pargs.get('structure', 'resnet18')
    scheme = pargs.get('scheme', 'bt')

    backbone = get_backbone(structure, weight=None)
    backbone.load_state_dict(get_substate(torch.load(f"{rdir}/models/{args.epoch}.pth", weights_only=True), 'backbone.'))
    if args.remove_last_relu:
        backbone[-2][-1].relu2 = nn.Identity()

    scheme_cls = scheme_name2class[scheme]
    model = scheme_cls.from_args(args, backbone)    
    transform = model.get_eval_transform()
    featurize_mtpc(f"{scheme}/{snameh}/{args.epoch}", args.num_workers, args['bsz'], backbone, transform)
