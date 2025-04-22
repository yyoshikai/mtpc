import sys, os, argparse, yaml, psutil, logging, math
from contextlib import nullcontext
import numpy as np, pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import lr_scheduler as lrs
from pl_bolts.optimizers import LARS
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/mtpc"]
from tools.path import make_result_dir
from tools.logger import get_logger, add_file_handler, add_stream_handler
from src.model import BarlowTwins, VICReg, VICRegL
from src.model.backbone import structures, get_backbone, structure2weights
from src.data.mtpc import MTPCRegionDataset, MTPCUHRegionDataset, MTPCVDRegionDataset
from src.data.image import TransformDataset
DDIR = f"{WORKDIR}/cheminfodata/mtpc"

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--studyname", default='default')
parser.add_argument('--data', nargs='+', default=['main'])
parser.add_argument('--nepoch', type=int, default=50)
## environment
parser.add_argument("--duplicate", default='ask')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--tqdm', action='store_true')
## model
parser.add_argument('--scheme', required=True)
parser.add_argument('--structure', choices=structures, required=True)
parser.add_argument('--weight')
args, _ = parser.parse_known_args()

if args.scheme == 'bt':
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--weight-decay', type=float, default=0)
    args, _ = parser.parse_known_args()
    if args.optimizer == 'lars':
        # 特にdefaultはないが, vicreg, vicreglのパラメータから判断して
        parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', default='cosine_annealing')

    parser.add_argument('--head-size', type=int, default=128)
    parser.add_argument('--lambda-param', type=float, default=5e-3)
    ## augmentation
    parser.add_argument('--resize-scale-min', type=float, default=0.08)
    parser.add_argument('--resize-scale-max', type=float, default=1.0)
    parser.add_argument('--resize-ratio-max', type=float, default=4/3)

elif args.scheme == 'vicreg':
    # --arch resnet50 --epochs 100
    parser.add_argument('--bsz', type=int, default=512)
    parser.add_argument('--base-lr', type=float, default=0.3)
    parser.add_argument('--optimizer', default='lars')
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    args, _ = parser.parse_known_args()
    if args.optimizer == 'lars':
        parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', default='constant')

    parser.add_argument('--sim-coeff', type=float, default=25.0)
    parser.add_argument('--std-coeff', type=float, default=25.0)
    parser.add_argument('--cov-coeff', type=float, default=1.0)
    parser.add_argument('--head-sizes', type=int, nargs='+', 
            default=[8192, 8192, 8192])

elif args.scheme == 'vicregl':

    parser.add_argument('--weight-decay', type=float, default=0.05)
    if 'convnext' in args.scheme:
        parser.add_argument('--bsz', type=int, default=384)
        parser.add_argument('--base-lr', type=float, default=0.00075)
        parser.add_argument('--optimizer', default='adamw')
    else:
        parser.add_argument('--bsz', type=int, default=512)
        parser.add_argument('--base-lr', type=float, default=0.3)
        parser.add_argument('--optimizer', default='lars')
    args, _ = parser.parse_known_args()
    if args.optimizer == 'lars':
        parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', default='cosine_annealing_warmup')
    
    parser.add_argument('--sim-coeff', type=float, default=25.0)
    parser.add_argument('--std-coeff', type=float, default=25.0)
    parser.add_argument('--cov-coeff', type=float, default=1.0)
    parser.add_argument('--head-sizes', type=int, nargs='+', 
            default=[8192, 8192, 8192])
    parser.add_argument('--map-head-sizes', type=int, nargs='+', 
            default=[512, 512, 512])
    parser.add_argument('--num-matches', type=int, nargs='+', default=[20, 4])
    parser.add_argument("--size-crops", type=int, nargs="+", default=[224, 96])
    parser.add_argument("--num-crops", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--min_scale_crops", type=float, nargs="+", default=[0.4, 0.08])
    parser.add_argument("--max_scale_crops", type=float, nargs="+", default=[1, 0.4])
    parser.add_argument('--alpha', default=0.75)
args, _ = parser.parse_known_args()
if args.scheduler == 'cosine_annealing_warmup':
    parser.add_argument('--warmup', type=int, default=10) # default in VICRegL

args = parser.parse_args()
if hasattr(args, 'base_lr'):
    args.lr = args.base_lr * args.bsz / 256

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

data = []
## main data
if 'main' in args.data:
    df_wsi = pd.read_csv(f"{DDIR}/processed/annotation_check0.csv", index_col=0, 
        dtype=str, keep_default_na=False)
    for wsi_name in df_wsi.index:
        for region_idx in df_wsi.columns:
            if df_wsi.loc[wsi_name, region_idx] == 'NaN':
                continue
            data.append(MTPCRegionDataset(wsi_name, region_idx, 256))
## additional data
if 'add' in args.data:
    for wsi_idx in range(1, 106):
        data += [MTPCUHRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
    for wsi_idx in range(1, 55):
        data += [MTPCVDRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
data = ConcatDataset(data)

# Model
weight = args.weight if args.weight in structure2weights[args.structure] else None
backbone = get_backbone(args.structure, weight)
match args.scheme:
    case 'bt':
        model = BarlowTwins(backbone, args.lambda_param, args.head_size, 
                args.resize_scale_min, args.resize_scale_max, args.resize_ratio_max)
    case 'vicreg':
        model = VICReg(backbone, args.head_sizes, args.sim_coeff, 
                args.std_coeff, args.cov_coeff)
    case 'vicregl':
        head_norm = 'batch_norm' if 'resnet' in args.structure else 'layer_norm'
        logger.info(f"{head_norm=}")

        model = VICRegL(backbone, args.head_sizes, args.map_head_sizes, args.alpha, 
                head_norm, args.sim_coeff, args.std_coeff, args.cov_coeff, True, 
                args.num_matches, False, args.size_crops, args.num_crops, 
                args.min_scale_crops, args.max_scale_crops, True)
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
    case 'cosine_annealing_warmup':
        def lr_lambda(epoch: int):

            if epoch <= 0:
                return 1.0
            elif epoch < args.warmup:
                return epoch / args.warmup
            else:
                end_scale = 0.001
                decay = 0.5*(1+math.cos(math.pi*(epoch-args.warmup)/(args.nepoch-args.warmup)))
                return end_scale+(1-end_scale)*decay            
        scheduler = lrs.LambdaLR(optimizer, lr_lambda)

## Load weight
if args.weight is not None and weight is None:
    state = torch.load(args.weight, weights_only=True)
    new_state = {}
    for k, v in state.items():
        if k.startswith('backbone.'):
            k = k[9:]
            new_state[k] = v
    logger.info(model.backbone.load_state_dict(new_state))

# Data augmentation
transform = model.get_train_transform(f"{rdir}/augment_example", 1)
data = TransformDataset(data, transform)
loader = DataLoader(data, batch_size=args.bsz, shuffle=True, 
    num_workers=args.num_workers, pin_memory=True)



# Training
mean_losses = []
for iepoch in range(args.nepoch):
    logger.info(f"Epoch {iepoch} started.")
    model.train()

    losses = []
    with tqdm(loader, dynamic_ncols=True) if args.tqdm else nullcontext() as pbar:
        for x in loader:
            optimizer.zero_grad()
            loss = model(x)
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
