import sys, os
from argparse import ArgumentParser
from glob import glob
WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path += [f'{WORKDIR}/mtpc', WORKDIR]
from src.model.backbone import structures
from src.utils import RANDOM_STATE

# Argument
parser = ArgumentParser()

## model
parser.add_argument("--fname")

parser.add_argument('--studyname')
parser.add_argument("--structure", choices=structures)

## split
parser.add_argument('--split', choices=['n_ak_bin', 'n_ak_bin_noout'], required=True)

## weight
parser.add_argument('--weight')

# training
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--n-epoch", type=int, default=30)
parser.add_argument("--num-workers", type=int, default=None)
parser.add_argument("--tqdm", action='store_true')
parser.add_argument("--compile", action='store_true')
parser.add_argument("--early-stop", type=int, default=10)
parser.add_argument('--save-steps', action='store_true')
parser.add_argument('--save-model', action='store_true')
parser.add_argument('--save-pred', action='store_true')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--seed', type=int)
args = parser.parse_args()
## set default args
from_feature = args.fname is not None
if not from_feature:
    assert args.structure is not None
if args.num_workers is None:
    args.num_workers = 1 if from_feature else 28
args.use_val = True

# First check whether or not to do training.
## Whether result exists
if from_feature:
    assert args.studyname is None
    result_dir = f"feature_mlp/{args.fname}/dyskeratosis/data_wsi/{args.split}/0"
else:
    assert args.studyname is not None
    result_dir = f"finetune/{args.studyname}/dyskeratosis/data_wsi/{args.split}/0"
if os.path.exists(f"{result_dir}/score.csv") \
        and ((not args.save_steps) or os.path.exists(f"{result_dir}/steps.csv")) \
        and ((not args.save_model) or (len(glob(f"{result_dir}/best_model_*.pth")) >= 1)) \
        and ((not args.save_pred) or os.path.exists(f"{result_dir}/preds.csv")):
    print(f"All result exists for {result_dir}")
    sys.exit()

# import
import yaml
import pandas as pd, numpy as np, torch
from torch.utils.data import StackDataset, Subset, DataLoader, ConcatDataset
from src.utils.logger import add_stream_handler, get_logger
from src.utils.path import make_dir
from src.utils import set_random_seed
from src.data import untuple_dataset 
from src.data.mtpc import MTPCUHRegionDataset, MTPCVDRegionDataset, MTPCDataset
from src.data.image import TransformDataset
from src.model import PredictModel, MLP
from src.model.backbone import get_backbone
from src.data import TensorDataset

# Environment
logger = get_logger()
add_stream_handler(logger)
## Directory
result_dir = make_dir(result_dir, 'overwrite')
with open(f"{result_dir}/args.yaml", 'w') as f:
    yaml.dump(vars(args), f)

## seed
set_random_seed(args.seed)
        
# Data
## target
y_train = pd.read_csv(f"{WORKDIR}/mtpc/data/target/patch.csv", index_col=0)['n_ak'].values
y_test = pd.read_csv(f"{WORKDIR}/mtpc/data/target/add_patch.csv", index_col=0)['dyskeratosis'].values

## mask
fold = np.load(f"{WORKDIR}/mtpc/data/split/main/wsi/{args.split}/0.npy")
train_mask = fold > 0
test_mask = np.isfinite(y_test)
y_train = y_train[train_mask]
y_test = y_test[test_mask]
train_target_data = TensorDataset(y_train)
test_target_data = TensorDataset(y_test)

## input
if from_feature:
    X_main = np.load(f"{WORKDIR}/mtpc/featurize/{args.fname}/feat_all.npy").astype(np.float32)
    train_input_data = TensorDataset(X_main[train_mask])

    X_add = np.load(f"{WORKDIR}/mtpc/featurize/{args.fname}/feat_added.npy").astype(np.float32)
    test_input_data = TensorDataset(X_add[test_mask])

    ### check nan
    nf_path = f"{result_dir}/nonfinite_params.txt"
    n_nf = 0
    with open(nf_path, 'w') as f:
        for param, name in zip([X_main, X_add, y_train, y_test], 
                ['X_train', 'X_test', 'y_train', 'y_test']):
            if np.any(~np.isfinite(param)):
                logger.warning(f"{name} contains nonfinite values.")
                f.write(name+'\n')
                n_nf += 1
    if n_nf > 0: 
        sys.exit()
    else:
        os.remove(nf_path)
    
else:    
    data = MTPCDataset(256)
    train_input_data, _ = untuple_dataset(data, 2)
    train_input_data = Subset(train_input_data, np.where(train_mask)[0])

    datas = []
    for wsi_idx in range(1, 106):
        datas += [MTPCUHRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
    for wsi_idx in range(1, 55):
        datas += [MTPCVDRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
    test_input_data = ConcatDataset(datas)
    test_input_data = Subset(test_input_data, np.where(test_mask)[0])


# model
output_mean = np.mean(y_train)
output_std = np.std(y_train)
if from_feature:
    assert args.structure is None
    input_size = X_main.shape[1]
    model = MLP(input_size, output_mean, output_std)
else:
    backbone = get_backbone(args.structure)
    model = PredictModel(backbone, output_mean, output_std)
    if args.weight is not None:
        state = torch.load(args.weight, weights_only=True)

        ## check nan
        nf_path = f"{result_dir}/nonfinite_params.txt"
        n_nf = 0
        with open(nf_path, 'w') as f:
            for name, param in state.items():
                if torch.any(~torch.isfinite(param)):
                    logger.warning(f"{name} contains nonfinite values.")
                    f.write(name+'\n')
                    n_nf += 1
        if n_nf > 0: 
            sys.exit()
        else:
            os.remove(nf_path)

        logger.info(model.backbone.load_state_dict(state))
    
    ## get transform
    transforms = backbone.get_transforms()
    train_input_data = TransformDataset(train_input_data, transforms)
    test_input_data = TransformDataset(test_input_data, transforms)

train_data = StackDataset(train_input_data, train_target_data)
test_data = StackDataset(test_input_data, test_target_data)

prefetch_factor = 5 if args.num_workers > 0 else None
print(f"{len(train_data)=}, {len(test_data)=}")

train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor, generator=torch.Generator())
test_loader = DataLoader(test_data, args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)

from src.predict import predict
predict(model, True, result_dir, args.n_epoch, args.early_stop, args.lr, 
    output_std, train_loader, test_loader, None, args.optimizer, args.compile, args.tqdm, 
    args.save_steps, args.save_pred, args.save_model)

