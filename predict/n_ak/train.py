"""
mainのデータで学習(n_ak), addのデータ(dyskeratosis)で評価
"""
import sys, os
import yaml
from argparse import ArgumentParser
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import StackDataset, Subset, DataLoader, ConcatDataset

WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path += [f'{WORKDIR}/mtpc', WORKDIR]
from src.utils import RANDOM_STATE
from src.utils.logger import add_stream_handler, get_logger
from src.utils.path import make_dir
from src.data import untuple_dataset 
from src.data.mtpc import MTPCUHRegionDataset, MTPCVDRegionDataset, MTPCDataset
from src.data.image import TransformDataset
from src.model import MLP, PredictModel
from src.model.backbone import get_backbone, structures, structure2weights
from src.data import TensorDataset

# args
parser = ArgumentParser()
parser.add_argument("--studyname", required=True)
parser.add_argument("--feature-name")
parser.add_argument("--structure", choices=structures)
parser.add_argument('--weight')
## training
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--n-epoch", type=int, default=30)
parser.add_argument("--num-workers", type=int, default=None)
parser.add_argument("--tqdm", action='store_true')
parser.add_argument("--compile", action='store_true')
parser.add_argument("--duplicate", default='ask')
parser.add_argument("--early-stop", type=int, default=10)
parser.add_argument('--save-steps', action='store_true')
parser.add_argument('--save-model', action='store_true')
parser.add_argument('--save-pred', action='store_true')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

## default args
from_feature = args.feature_name is not None
if not from_feature:
    assert args.structure is not None
if args.num_workers is None:
    args.num_workers = 1 if from_feature else 28

# Environment
logger = get_logger()
add_stream_handler(logger)
RANDOM_STATE.seed(args.seed)


# Data
## df
df = pd.read_csv(f"{WORKDIR}/mtpc/data/target/patch.csv", index_col=0)
dfa = pd.read_csv(f"{WORKDIR}/mtpc/data/target/add_patch.csv", index_col=0)
a_mask = np.isfinite(dfa['dyskeratosis'].values)
dfa = dfa[a_mask]

## y
y = df['n_ak'].values
y_add = dfa['dyskeratosis'].values
train_target_data = TensorDataset(y)
test_target_data = TensorDataset(y_add)

## X
if from_feature:
    X = np.load(f"{WORKDIR}/mtpc/featurize/{args.feature_name}/feat_all.npy").astype(np.float32)
    train_input_data = TensorDataset(X)
    X_add = np.load(f"{WORKDIR}/mtpc/featurize/{args.feature_name}/feat_added.npy").astype(np.float32)[a_mask]
    test_input_data = TensorDataset(X_add)
else:
    data = MTPCDataset(256)
    train_input_data, _ = untuple_dataset(data, 2)
    datas = []
    for wsi_idx in range(1, 55):
        datas += [MTPCVDRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
    test_input_data = ConcatDataset(datas)

# model
output_mean = np.mean(y)
output_std = np.std(y)
if from_feature:
    assert args.structure is None
    input_size = X.shape[1]
    model = MLP(input_size, output_mean, output_std)
else:
    use_weight = args.weight in structure2weights[args.structure] and args.weight is not None
    backbone = get_backbone(args.structure, args.weight if use_weight else None)
    model = PredictModel(backbone, output_mean, output_std)
    if not use_weight and args.weight is not None:
        whole_state: dict[str, torch.Tensor]
        whole_state = torch.load(f"{WORKDIR}/mtpc/pretrain/{args.weight}", weights_only=True)
        state = {}
        for key, value in whole_state.items():
            if key.startswith('backbone.'):
                state[key[9:]] = value
        model.backbone.__delattr__
        logger.info(model.backbone.load_state_dict(state))
    
    ## get transform
    transforms = backbone.get_transforms()
    train_input_data = TransformDataset(train_input_data, transforms)
    test_input_data = TransformDataset(test_input_data, transforms)

## loader
prefetch_factor = 5 if args.num_workers > 0 else None
train_data = StackDataset(train_input_data, train_target_data)
train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
print(len(test_input_data), len(test_target_data))
test_data = StackDataset(test_input_data, test_target_data)
test_loader = DataLoader(test_data, args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)

# Directory
result_dir = f"./results/{args.studyname}"
result_dir = make_dir(result_dir, args.duplicate)
with open(f"{result_dir}/args.yaml", 'w') as f:
    yaml.dump(vars(args), f)

from src.predict import predict
predict(model, True, result_dir, args.n_epoch, args.early_stop, output_std, 
        train_loader, test_loader, None, args.compile, args.tqdm, 
        args.save_steps, args.save_pred, args.save_model)

