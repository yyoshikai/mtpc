from argparse import ArgumentParser

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
parser.add_argument("--use-val", action='store_true', help="If True, use validation set for training. Final model is evaluated as best model.")
parser.add_argument("--from-scratch", action='store_true')
parser.add_argument("--structure", default='resnet50')
parser.add_argument('--init-weight', help='set scratch to train from initial' \
        'model, or set path from .../pretrain, defaults to imagenet')
args = parser.parse_args()
## set default args
from_feature = args.feature_name is not None
if args.num_workers is None:
    args.num_workers = 1 if from_feature else 28
if from_feature:
    assert args.init_weight is None
else:
    if args.init_weight is None:
        args.init_weight = 'imagenet'

# First check whether or not to do training.
## Whether result exists
import sys, os

result_dir = f"./results/{args.studyname}/{args.target}/{args.split}"
if os.path.exists(f"{result_dir}/score.csv"):
    print(f"{result_dir}/score.csv already exists.")
    sys.exit()

## fold exists?
WORKDIR = os.environ.get('WORKDIR', '/workspace')
fold_path = f"{WORKDIR}/mtpc/data/split/{'add' if args.add else 'main'}/{args.split}.npy"
if not os.path.exists(fold_path):
    print(f"{fold_path=} does not exist.")
    sys.exit()

## fold is valid?
import numpy as np, pandas as pd
sys.path += [f'{WORKDIR}/mtpc', WORKDIR]
from src.predict import get_mask

folds = np.load(fold_path)
df = pd.read_csv(f"{WORKDIR}/mtpc/data/target/{'add_patch' if args.add else 'patch'}.csv", index_col=0)
y = df[args.target].values
train_mask, val_mask, test_mask = get_mask(y, folds, args.reg)
if train_mask is None:
    print(f"folds cannot split y correctly.")
    sys.exit()


# import
import yaml, math
from tqdm import tqdm
import pandas as pd, torch, torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import StackDataset, Subset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error, mean_absolute_error
from src.utils.logger import add_stream_handler, add_file_handler, get_logger
from src.utils.path import make_dir
from src.data import untuple_dataset, MTPCDataset, BaseAugmentDataset
from src.data.mtpc import MTPCUHRegionDataset, MTPCVDRegionDataset
from src.model import ResNetModel as Model
from src.data import TensorDataset

# Environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger()
add_stream_handler(logger)

# Data
if args.add:
    if from_feature:
        X = np.load(f"{WORKDIR}/mtpc/featurize/{args.feature_name}/feat_added.npy")
        input_data = TensorDataset(X)
    else:
        datas = []
        for wsi_idx in range(1, 106):
            datas += [MTPCUHRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
        for wsi_idx in range(1, 55):
            datas += [MTPCVDRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
        input_data = ConcatDataset(datas)
        input_data = BaseAugmentDataset(input_data)
else:
    if from_feature:
        X = np.load(f"{WORKDIR}/mtpc/featurize/{args.feature_name}/feat_all.npy")
        input_data = TensorDataset(X)
    else:
        data = MTPCDataset(256)
        input_data, _ = untuple_dataset(data, 2)
        input_data = BaseAugmentDataset(input_data)
target_data = TensorDataset(y)
data = StackDataset(input_data, target_data)

prefetch_factor = 5 if args.num_workers > 0 else None
if args.use_val:
    train_data = Subset(data, np.where(train_mask|val_mask)[0])
    test_data = Subset(data, np.where(test_mask)[0])
else:
    train_data = Subset(data, np.where(train_mask)[0])
    val_data = Subset(data, np.where(val_mask)[0])
    test_data = Subset(data, np.where(test_mask)[0])
    val_loader = DataLoader(val_data, args.batch_size*2, False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
loader = DataLoader(train_data, args.batch_size, True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)

# Directory
result_dir = make_dir(result_dir, args.duplicate)
os.makedirs(f"{result_dir}/models", exist_ok=True)
add_file_handler(logger, f"{result_dir}/train.log")
with open(f"{result_dir}/args.yaml", 'w') as f:
    yaml.dump(vars(args), f)

if from_feature:
    class MLP(nn.Sequential):
        def __init__(self):
            super().__init__(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        def forward(self, input):
            return super().forward(input).squeeze(-1)
    model = MLP()
else:
    model = Model(args.structure, from_scratch=args.init_weight == 'scratch')
    if args.init_weight not in ['imagenet', 'scratch']:
        whole_state: dict[str, torch.Tensor]
        whole_state = torch.load(f"{WORKDIR}/mtpc/pretrain/{args.init_weight}", weights_only=True)
        state = {}
        for key, value in whole_state.items():
            if key.startswith('backbone.'):
                state[key[9:]] = value
        logger.info(model.backbone.load_state_dict(state))

model.to(device)
if args.compile:
    model = torch.compile(model)
if args.reg:
    criterion = nn.MSELoss(reduction='mean')
else:
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters())
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Train
context = tqdm if args.tqdm else lambda x: x
losses = []
score_path = f"{result_dir}/val_scores.csv"
with open(score_path, 'w') as f:
    if args.reg:
        f.write("epoch,R^2,RMSE,MAE\n")
    else:
        f.write("epoch,AUROC,AUPR")

best_score = -math.inf
best_epoch = None
for epoch in range(args.n_epoch):
    
    model.train()
    for input_batch, target_batch in context(loader):
        optimizer.zero_grad()
        pred_batch = model(input_batch.to(device))
        loss = criterion(pred_batch, target_batch.to(torch.float).to(device))
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    scheduler.step()
    epoch += 1   

    ## Save steps
    df = pd.DataFrame({'loss': losses})
    df.to_csv(f"{result_dir}/steps.csv", index_label='step')

    ## Evaluate
    if args.use_val:
        if epoch == args.n_epoch:
            torch.save(model.state_dict(), f"{result_dir}/models/{epoch}.pth")
    else:
        logger.info(f"Evaluating epoch {epoch}...")
        model.eval()
        preds = []
        targets = []
        with torch.inference_mode():
            for input_batch, target_batch in context(val_loader):
                pred_batch = model(input_batch.to(device))
                preds.append(pred_batch.cpu().numpy())
                targets.append(target_batch.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        with open(score_path, 'a') as f:
            if args.reg:
                score = -mean_squared_error(targets, preds)
                f.write(f"{epoch},{r2_score(targets,preds)},{mean_squared_error(targets, preds)**0.5},{mean_absolute_error(targets, preds)}\n")
            else:
                targets = targets.astype(int)
                f.write(f"{epoch},{roc_auc_score(targets, preds)},{average_precision_score(targets, preds)}\n")
                score = roc_auc_score(targets, preds)

        ## Early stopping
        if best_score < score:
            if best_epoch is not None:
                os.remove(f"{result_dir}/models/{best_epoch}.pth")
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), f"{result_dir}/models/{epoch}.pth")
        else:
            if epoch - best_epoch >= args.early_stop:
                break
# Evaluate for test data
logger.info(f"Evaluating for test_data with best model ({best_epoch})...")
test_loader = DataLoader(test_data, args.batch_size*2, False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
if not args.use_val:
    model.load_state_dict(torch.load(f"{result_dir}/models/{best_epoch}.pth", weights_only=True))
model.eval()
preds = []
targets = []
with torch.inference_mode():
    for input, target in context(test_loader):
        pred = model(input.to(device))
        preds.append(pred.cpu().numpy())
        targets.append(target.numpy())
preds = np.concatenate(preds)
targets = np.concatenate(targets)
if args.reg:
    df = pd.DataFrame({'score': {
        'RMSE': mean_squared_error(targets, preds)**0.5,
        'MAE': mean_absolute_error(targets, preds),
        'R^2': r2_score(targets, preds)
    }})
else:
    targets = targets.astype(int)
    df = pd.DataFrame({'score': {
        'AUROC': roc_auc_score(targets, preds),
        'AUPR': average_precision_score(targets, preds),
    }})
df.to_csv(f"{result_dir}/score.csv")

logger.info(f"training {args.studyname}/{args.target}/{args.split} finished!")
