import sys, os, argparse, yaml, psutil, logging
from contextlib import nullcontext
import numpy as np, pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, ConcatDataset, StackDataset
from torch.optim import lr_scheduler as lrs
from pl_bolts.optimizers import LARS
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/mtpc"]
from tools.path import make_result_dir, timestamp
from tools.logger import get_logger, add_file_handler, add_stream_handler
from src.model import BarlowTwins, VICReg
from src.model.backbone import structures, get_backbone, structure2weights
from src.data import untuple_dataset, ShuffleAugmentDataset, AugmentDataset, CacheDataset
from src.data.mtpc import MTPCRegionDataset, MTPCUHRegionDataset, MTPCVDRegionDataset

DDIR = f"{WORKDIR}/cheminfodata/mtpc"

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--studyname", default='default')
parser.add_argument("--duplicate", default='ask')

parser.add_argument('--lambda-param', type=float, default=5e-3)
parser.add_argument('--shuffle-aug', choices=['region', 'wsi'], default=None)
parser.add_argument('--data', nargs='+', default=['main'])
# parser.add_argument('--add-data', action='store_true')

## model
parser.add_argument('--scheme', required=True)
parser.add_argument('--structure', choices=structures, default='resnet18')
parser.add_argument('--weight')
### Barlow Twins
parser.add_argument('--head-size', type=int, default=128)
### VICReg
# TODO: base_lr=0.3, LARS optimizer
parser.add_argument('--sim-coeff', type=float, default=25.0)
parser.add_argument('--std-coeff', type=float, default=25.0)
parser.add_argument('--cov-coeff', type=float, default=1.0)
parser.add_argument('--head-sizes', type=int, nargs='+', 
        default=[8192, 8192, 8192])

## training
parser.add_argument('--bsz', type=int)
parser.add_argument('--nepoch', type=int, default=50)

## optimizer
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--weight-decay', type=float)
parser.add_argument('--momentum', type=float, default=0.9) # lars
parser.add_argument('--scheduler', choices=['constant', 'cosine_annealing'], 
        default='cosine_annealing')

## augmentation
parser.add_argument('--resize-scale-min', type=float, default=0.08)
parser.add_argument('--resize-scale-max', type=float, default=1.0)
parser.add_argument('--resize-ratio-max', type=float, default=4/3)

## environment
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--tqdm', action='store_true')
args = parser.parse_args()
if args.weight == 'resnet':
    from_resnet = True
    args.weight = None
else:
    from_resnet = False

if args.scheme == 'bt':
    args.bsz = args.bsz or 64
elif args.scheme == 'vicreg':
    args.bsz = args.bsz or 512 # from github

if args.weight_decay is None:
    if args.optimizer == 'adam':
        args.weight_decay = 0
    elif args.optimizer == 'lars':
        args.weight_decay = 1e-6 # default in VICReg. 0だとLARSの効果がなくなる。
    else:
        raise ValueError

# Environment
rdir = make_result_dir(dirname=f"./results/{args.studyname}", duplicate=args.duplicate)
os.makedirs(f"{rdir}/models")

with open(f"{rdir}/args.yaml", 'w') as f:
    yaml.dump(vars(args), f)

## logger
logger = get_logger()
add_file_handler(logger, f'{rdir}/info.log', level=logging.INFO)
add_file_handler(logger, f'{rdir}/debug.log', level=logging.DEBUG)
add_stream_handler(logger)

## device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"device={device}")

# Data
random_crop_size = (224, 224)
color_plob, blur_plob, solar_plob = 0.8, 0.4, 0.0

data0 = []
## main data
if 'main' in args.data:
    df_wsi = pd.read_csv(f"{DDIR}/processed/annotation_check0.csv", index_col=0, 
        dtype=str, keep_default_na=False)
    for wsi_name in df_wsi.index:
        region_datas0 = []
        for region_idx in df_wsi.columns:
            if df_wsi.loc[wsi_name, region_idx] == 'NaN':
                continue
            region_datas0.append(MTPCRegionDataset(wsi_name, region_idx, 256))
        data0.append(region_datas0)
## additional data
if 'add' in args.data:
    for wsi_idx in range(1, 106):
        data0.append([MTPCUHRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)])
    for wsi_idx in range(1, 55):
        data0.append([MTPCVDRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)])
## add shuffle augmentation
match args.shuffle_aug:
    case 'region':
        data = ConcatDataset([
            ConcatDataset([ShuffleAugmentDataset(region_data) for region_data in region_datas]) for region_datas in data0
        ])
        data0, data1 = untuple_dataset(data, 2)
    case 'wsi':
        data = ConcatDataset([
            ShuffleAugmentDataset(ConcatDataset[region_datas]) for region_datas in data0
        ])
        data0, data1 = untuple_dataset(data, 2)
    case _:
        data = ConcatDataset([ ConcatDataset(region_datas) for region_datas in data0 ])
        data0 = data1 = CacheDataset(data)
data = StackDataset(
    *[AugmentDataset(data, color_plob, blur_plob, solar_plob, random_crop_size,
        args.resize_scale_min, args.resize_scale_max, args.resize_ratio_max,
        f"{rdir}/augment_example/{i}", 1) for i, data in enumerate([data0, data1])]
)
loader = DataLoader(data, batch_size=args.bsz, shuffle=True, 
    num_workers=args.num_workers, pin_memory=True)

# Model
weight = args.weight if args.weight in structure2weights[args.structure] else None
backbone = get_backbone(args.structure, weight)
match args.scheme:
    case 'bt':
        model = BarlowTwins(from_resnet=from_resnet, head_size=args.head_size)
    case 'vicreg':
        model = VICReg(backbone, args.head_sizes, args.sim_coeff, 
                args.std_coeff, args.cov_coeff)
    case _:
        raise ValueError
model.to(device)
match args.optimizer:
    case 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    case 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, momentum=args.momentum, 
                weight_decay=args.weight_decay)
    case _: raise ValueError
match args.scheduler:
    case 'constant':
        scheduler = lrs.ConstantLR(optimizer, factor=1.0)
    case 'cosine_annealing':
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=args.nepoch, eta_min=0)

## Load weight
if args.weight is not None and weight is None:
    state = torch.load(args.weight, weights_only=True)
    new_state = {}
    for k, v in state.items():
        if k.startswith('backbone.'):
            k = k[9:]
            new_state[k] = v
    logger.info(model.backbone.load_state_dict(new_state))

# Training
mean_losses = []
for iepoch in range(args.nepoch):
    logger.info(f"Epoch {iepoch} started.")
    model.train()

    losses = []
    with tqdm(loader, dynamic_ncols=True) if args.tqdm else nullcontext() as pbar:
        for x_a, x_b in loader:
            optimizer.zero_grad()
            loss = model(x_a.to(device), x_b.to(device))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if args.tqdm:
                pbar.update()
                mem = psutil.virtual_memory()
                pbar.postfix = f"{mem.used/2**30:.03f}/{mem.total/2**30:.03f}"
    mean_loss = np.mean(losses)
    mean_losses.append(mean_loss)
    df_epoch = pd.DataFrame({'loss': mean_losses})
    df_epoch.to_csv(f"{rdir}/epoch.csv", index_label='Epoch')
    torch.save(model.state_dict(), f"{rdir}/models/{iepoch}.pth")
    scheduler.step()

    logger.info(f"Epoch {iepoch} ended.")
