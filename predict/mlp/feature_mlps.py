import sys, os
from glob import glob
from argparse import ArgumentParser

# Argument
parser = ArgumentParser()

## target
parser.add_argument('--target', required=True)
parser.add_argument('--split', required=True)
parser.add_argument('--add', action='store_true')
parser.add_argument('--reg', action='store_true')

## feature
parser.add_argument('--feature-names', nargs='+')

## training
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--n-epoch", type=int, default=30)
parser.add_argument("--tqdm", action='store_true')
parser.add_argument("--early-stop", type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optimizer', default='adam')
args = parser.parse_args()
args.use_val = True # Fix for check fold is valid

# First check whether or not to do training.
## Whether result exists
fnames = []
for fname in args.feature_name:
    result_dir = f"feature_mlp/{fname}/{args.target}/{args.split}"
    if os.path.exists(f"{result_dir}/score.csv") \
            and ((not args.save_steps) or os.path.exists(f"{result_dir}/steps.csv")) \
            and ((not args.save_model) or (len(glob(f"{result_dir}/best_model_*.pth")) >= 1)) \
            and ((not args.save_pred) or os.path.exists(f"{result_dir}/preds.csv")):
        print(f"All result exists for {fname}")
    else:
        fnames.append(fname)
if len(fnames) == 0:
    print("All results exist.")
    sys.exit()
## fold exists?
WORKDIR = os.environ.get('WORKDIR', '/workspace')
fold_path = f"{WORKDIR}/mtpc/data/split/{'add' if args.add else 'main'}/{args.split}.npy"
if not os.path.exists(fold_path):
    print(f"{fold_path=} does not exist.")
    sys.exit()
## fold is valid?
import numpy as np, pandas as pd
df = pd.read_csv(f"{WORKDIR}/mtpc/data/target/{'add_patch' if args.add else 'patch'}.csv", index_col=0)
y = df[args.target].values
folds = np.load(fold_path)
test_mask = folds == 0
train_mask = folds > 0
y_test = y[test_mask]
y_train = y[train_mask]
if (not args.reg) and (np.all(y_test == y_test[0]) or np.all(y_train == y_train[0])):
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
from src.model import fill_output_mean_std
from src.model.backbone import get_backbone, structure2weights
from src.data import TensorDataset

# Environment
logger = get_logger()
add_stream_handler(logger)
        
# Data
npy_name = 'feat_added' if args.add else 'feat_all'
X = [np.load(f"{WORKDIR}/mtpc/featurize/{fname}/{npy_name}.npy").astype(np.float32)
        for fname in fnames]
X = np.concateante(X, axis=1) # [B, F, D] F:feature
input_data = TensorDataset(X)

# model
if args.reg:
    output_mean = np.mean(y[train_mask])
    output_std = np.std(y[train_mask])
else:
    output_mean = 0.0
    output_std = 1.0

# ここから未実装
class MultiMLP(nn.Sequential):
    def __init__(self, input_size, output_mean: float=0.0, output_std: float=1.0):
        super().__init__(nn.Linear(input_size, 128), nn.GELU(), nn.Linear(128, 1))
        self.register_buffer('output_mean', torch.tensor(output_mean))
        self.register_buffer('output_std', torch.tensor(output_std))
        self.register_load_state_dict_post_hook(fill_output_mean_std)
    def forward(self, input):
        return super().forward(input).squeeze(-1)*self.output_std+self.output_mean


input_size = X.shape[-1]
model = MLP(input_size, output_mean, output_std)

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







