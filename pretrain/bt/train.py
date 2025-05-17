import sys, os, argparse, yaml, psutil, logging, math
from contextlib import nullcontext
from logging import getLogger
import numpy as np, pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import lr_scheduler as lrs
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/mtpc"]
from tools.path import make_result_dir
from tools.logger import get_logger, add_file_handler, add_stream_handler
from src.model.backbone import structures, get_backbone, structure2weights
from src.model.scheme import scheme_name2class
from src.optimizer.lars import LARS
from src.data.mtpc import MTPCRegionDataset, MTPCUHRegionDataset, MTPCVDRegionDataset
from src.data.image import TransformDataset
from src.utils.utils import logend

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
parser.add_argument('--fp16', action='store_true')
## model
parser.add_argument('--scheme', required=True)
parser.add_argument('--structure', choices=structures, required=True)
parser.add_argument('--weight')
args, _ = parser.parse_known_args()

scheme_cls = scheme_name2class[args.scheme]
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
scheme_cls.add_args(parser)

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
getLogger('tifffile').disabled = True
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
model = scheme_cls.from_args(args, backbone)
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
loader_size = math.ceil(len(data)/args.bsz)


# Training
mean_losses = []
for iepoch in range(args.nepoch):
    logger.info(f"Epoch {iepoch} started.")
    model.train()

    losses = []
    data_iter = loader.__iter__()
    if args.tqdm:
        pbar = tqdm(total=loader_size, dynamic_ncols=True)
    step = 0
    while True:
        try:
            with logend(logger, 'load_data') if step == 0 else nullcontext():
                x = data_iter.__next__()
        except StopIteration:
            break
        optimizer.zero_grad()
        with torch.autocast('cuda', torch.bfloat16) if args.fp16 else nullcontext():
            loss = model(x)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if args.tqdm:
            pbar.update()
            mem = psutil.virtual_memory()
            pbar.postfix = f"{mem.used/2**30:.03f}GB/{mem.total/2**30:.03f}GB"
        step += 1
    mean_loss = np.mean(losses)
    mean_losses.append(mean_loss)
    df_epoch = pd.DataFrame({'loss': mean_losses})
    df_epoch.to_csv(f"{rdir}/epoch.csv", index_label='Epoch')
    torch.save(model.state_dict(), f"{rdir}/models/{iepoch}.pth")
    scheduler.step()

    logger.info(f"Epoch {iepoch} ended.")
