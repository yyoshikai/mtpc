import sys, os, yaml
from argparse import ArgumentParser
import numpy as np, pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, StackDataset, ConcatDataset, \
        TensorDataset, Subset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, \
        roc_auc_score, average_precision_score
from addict import Dict
WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path += [f'{WORKDIR}/mtpc', WORKDIR]
from src.data import untuple_dataset, MTPCDataset, BaseAugmentDataset, InDataset
from src.data.mtpc import MTPCUHRegionDataset, MTPCVDRegionDataset
from src.model import ResNetModel as Model
LABEL_TYPES = ['Other', 'Normal', 'Mild', 'Moderate', 'Severe']


parser = ArgumentParser()
parser.add_argument('--studyname', required=True)
parser.add_argument('--tqdm', action='store_true')
parser.add_argument('--num-workers', type=int, default=28)
args = parser.parse_args()

sdir = f"results/{args.studyname}"
with open(f"{sdir}/args.yaml") as f:
    sargs = Dict(yaml.safe_load(f))

# Environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
context = tqdm if args.tqdm else lambda x: x

# Data
if sargs.target == 'find':
    data = MTPCDataset(256)
    image_data, label_data = untuple_dataset(data, 2)
    image_data = BaseAugmentDataset(image_data)
    label_data = InDataset(label_data, LABEL_TYPES[LABEL_TYPES.index(sargs.positive_th):])
    data = StackDataset(image_data, label_data)
    split_dir = f"./split/results/{sargs.split}"
else:
    datas = []
    for wsi_idx in range(1, 106):
        datas += [MTPCUHRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
    for wsi_idx in range(1, 55):
        datas += [MTPCVDRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
    data = ConcatDataset(datas)
    data = BaseAugmentDataset(data)
    df = pd.read_csv(f"{WORKDIR}/mtpc/cnn/split/add_patch.csv", index_col=0)
    label_data = TensorDataset(torch.Tensor(df[sargs.target]))
    data = StackDataset(data, label_data)
    split_dir = f"./split/add/{sargs.split}"
test_data = Subset(data, np.load(f"{split_dir}/test.npy"))
loader = DataLoader(test_data, sargs.batch_size, True, num_workers=args.num_workers,
        pin_memory=True, prefetch_factor=5, persistent_workers=False)

# Model
model = Model()
model.to(device)
model.eval()
## Get best model
dfscore = pd.read_csv(f"{sdir}/score.csv", index_col=0)
if sargs.reg:
    epoch = dfscore.index.values[np.argmin(dfscore['RMSE'].values)]
else:
    epoch = dfscore.index.values[np.argmax(dfscore['AUROC'].values)]
model.load_state_dict(torch.load(f"{sdir}/models/{epoch}.pth", weights_only=True))


preds = []
labels = []
with torch.inference_mode():
    for i, (image_batch, label_batch) in enumerate(context(loader)):
        if sargs.add:
            label_batch = label_batch[0]
        pred_batch = model(image_batch.to(device))
        preds.append(pred_batch.cpu().numpy())
        labels.append(label_batch.numpy())
preds = np.concatenate(preds)
labels = np.concatenate(labels)
if sargs.reg:
    score = {
        'R^2': r2_score(labels, preds),
        'RMSE': mean_squared_error(labels, preds)**0.5,
        'MAE': mean_absolute_error(labels, preds)
    }
else:
    labels = labels.astype(int)
    score = {
        'AUROC': roc_auc_score(labels, preds),
        'AUPR': average_precision_score(labels, preds)
    }
pd.DataFrame({epoch: score}).to_csv(f"{sdir}/test_score.csv")
