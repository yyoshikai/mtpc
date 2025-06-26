import sys, os
from argparse import ArgumentParser
from glob import glob
WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path += [f'{WORKDIR}/mtpc', WORKDIR]
from src.model.backbone import structures
from src.model.state_dict import state_modifiers

# Argument
parser = ArgumentParser()
parser.add_argument("--studyname", required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--add", action='store_true')
parser.add_argument("--target", required=True)
parser.add_argument("--reg", action='store_true')
parser.add_argument("--feature-name")

# training
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--n-epoch", type=int, default=30)
parser.add_argument("--num-workers", type=int, default=None)
parser.add_argument("--tqdm", action='store_true')
parser.add_argument("--compile", action='store_true')
parser.add_argument("--duplicate", default='ask')
parser.add_argument("--early-stop", type=int, default=10)
parser.add_argument("--use-val", action='store_true', help="If True, use validation set for training. The model is evaluated by the final model.")
parser.add_argument("--structure", choices=structures)
parser.add_argument('--weight')
parser.add_argument('--state-modifier', default='bt')
parser.add_argument('--save-steps', action='store_true')
parser.add_argument('--save-model', action='store_true')
parser.add_argument('--save-pred', action='store_true')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optimizer', default='adam')
args = parser.parse_args()
## set default args
from_feature = args.feature_name is not None
if not from_feature:
    assert args.structure is not None
if args.num_workers is None:
    args.num_workers = 1 if from_feature else 28

# First check whether or not to do training.
## Whether result exists
result_dir = f"./{args.studyname}/{args.target}/{args.split}"
if os.path.exists(f"{result_dir}/score.csv") \
        and ((not args.save_steps) or os.path.exists(f"{result_dir}/steps.csv")) \
        and ((not args.save_model) or (len(glob(f"{result_dir}/best_model_*.pth")) >= 1)) \
        and ((not args.save_pred) or os.path.exists(f"{result_dir}/preds.csv")):
    print(f"All result exists for {result_dir}")
    sys.exit()
## fold exists?
fold_path = f"{WORKDIR}/mtpc/data/split/{'add' if args.add else 'main'}/{args.split}.npy"
if not os.path.exists(fold_path):
    print(f"{fold_path=} does not exist.")
    sys.exit()

## fold is valid?
import numpy as np, pandas as pd
from src.predict import get_mask

folds = np.load(fold_path)
df = pd.read_csv(f"{WORKDIR}/mtpc/data/target/{'add_patch' if args.add else 'patch'}.csv", index_col=0)
y = df[args.target].values
train_mask, val_mask, test_mask = get_mask(y, folds, args.reg)
if train_mask is None:
    print(f"folds cannot split y correctly.")
    sys.exit()

# import
import yaml
import pandas as pd, torch, torch.nn as nn
from torch.utils.data import StackDataset, Subset, DataLoader, ConcatDataset
from src.utils.logger import add_stream_handler, get_logger
from src.utils.path import make_dir
from src.data import untuple_dataset 
from src.data.mtpc import MTPCUHRegionDataset, MTPCVDRegionDataset, MTPCDataset
from src.data.image import TransformDataset
from src.model import PredictModel, MLP
from src.model.backbone import get_backbone, structure2weights
from src.data import TensorDataset

# Environment
logger = get_logger()
add_stream_handler(logger)
        
# Data
if args.add:
    if from_feature:
        X = np.load(f"{WORKDIR}/mtpc/featurize/{args.feature_name}/feat_added.npy").astype(np.float32)
        input_data = TensorDataset(X)
    else:
        datas = []
        for wsi_idx in range(1, 106):
            datas += [MTPCUHRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
        for wsi_idx in range(1, 55):
            datas += [MTPCVDRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
        input_data = ConcatDataset(datas)
else:
    if from_feature:
        X = np.load(f"{WORKDIR}/mtpc/featurize/{args.feature_name}/feat_all.npy").astype(np.float32)
        input_data = TensorDataset(X)
    else:
        data = MTPCDataset(256)
        input_data, _ = untuple_dataset(data, 2)


# model
if args.reg:
    output_mean = np.mean(y[train_mask])
    output_std = np.std(y[train_mask])
else:
    output_mean = 0.0
    output_std = 1.0
if from_feature:
    assert args.structure is None
    input_size = X.shape[1]
    model = MLP(input_size, output_mean, output_std)
else:
    use_weight = args.weight in structure2weights[args.structure] and args.weight is not None
    backbone = get_backbone(args.structure, args.weight if use_weight else None)
    model = PredictModel(backbone, output_mean, output_std)
    if not use_weight and args.weight is not None:
        # environment to load state dict
        def exclude_bias_and_norm(p):
            return p.ndim == 1
        sys.path.append('/workspace/mtpc/pretrain/VICRegL')
        state = torch.load(f"{WORKDIR}/mtpc/pretrain/{args.weight}")
        state = state_modifiers[args.state_modifier](state)
        logger.info(model.backbone.load_state_dict(state))
    
    ## get transform
    transforms = backbone.get_transforms()
    input_data = TransformDataset(input_data, transforms)

target_data = TensorDataset(y)
data = StackDataset(input_data, target_data)

prefetch_factor = 5 if args.num_workers > 0 else None
if args.use_val:
    train_data = Subset(data, np.where(train_mask|val_mask)[0])
    test_data = Subset(data, np.where(test_mask)[0])
    val_loader = None
else:
    train_data = Subset(data, np.where(train_mask)[0])
    val_data = Subset(data, np.where(val_mask)[0])
    test_data = Subset(data, np.where(test_mask)[0])
    val_loader = DataLoader(val_data, args.batch_size*2, False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
print(f"{len(train_data)=}, {len(test_data)=}")
train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
test_loader = DataLoader(test_data, args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
# Directory
result_dir = make_dir(result_dir, args.duplicate)
with open(f"{result_dir}/args.yaml", 'w') as f:
    yaml.dump(vars(args), f)

from src.predict import predict
predict(model, args.reg, result_dir, args.n_epoch, args.early_stop, args.lr, 
    output_std, train_loader, test_loader, val_loader, args.optimizer, args.compile, args.tqdm, 
    args.save_steps, args.save_pred, args.save_model)

