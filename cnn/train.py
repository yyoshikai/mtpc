import sys, os, argparse, yaml, math
from tqdm import tqdm
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import StackDataset, Subset, DataLoader, ConcatDataset, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error, mean_absolute_error
WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path += [f'{WORKDIR}/mtpc', WORKDIR]
from src.utils.logger import add_stream_handler, add_file_handler, get_logger
from src.utils.path import make_dir
from src.data import untuple_dataset, MTPCDataset, BaseAugmentDataset, InDataset
from src.data.mtpc import MTPCUHRegionDataset, MTPCVDRegionDataset
from src.model import ResNetModel as Model
LABEL_TYPES = ['Other', 'Normal', 'Mild', 'Moderate', 'Severe']

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--studyname", required=True)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--split", required=True)
parser.add_argument("--n-epoch", type=int, default=20)
parser.add_argument("--positive-th", choices=LABEL_TYPES[1:])
parser.add_argument("--num-workers", type=int, default=28)
parser.add_argument("--tqdm", action='store_true')
parser.add_argument("--compile", action='store_true')
parser.add_argument("--duplicate", default='ask')
parser.add_argument("--add", action='store_true')
parser.add_argument("--target", choices=['bio', 'acantholysis', 'dyskeratosis'], 
        help='Ignored when not --add')
parser.add_argument("--early-stop", type=int, default=5)
parser.add_argument("--reg", action='store_true')
args = parser.parse_args()

# Environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger()
add_stream_handler(logger)

# Directory
result_dir = make_dir(f"./results/{args.studyname}", args.duplicate)
os.makedirs(f"{result_dir}/models", exist_ok=True)
add_file_handler(logger, f"{result_dir}/train.log")
with open(f"{result_dir}/args.yaml", 'w') as f:
    yaml.dump(vars(args), f)

# Data

if args.target == 'find':
    data = MTPCDataset(256)
    image_data, label_data = untuple_dataset(data, 2)
    image_data = BaseAugmentDataset(image_data)
    label_data = InDataset(label_data, LABEL_TYPES[LABEL_TYPES.index(args.positive_th):])
    data = StackDataset(image_data, label_data)
    split_dir = f"./split/results/{args.split}"
else:
    datas = []
    for wsi_idx in range(1, 106):
        datas += [MTPCUHRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
    for wsi_idx in range(1, 55):
        datas += [MTPCVDRegionDataset(wsi_idx, region_idx) for region_idx in range(1, 4)]
    data = ConcatDataset(datas)
    data = BaseAugmentDataset(data)
    df = pd.read_csv(f"{WORKDIR}/mtpc/cnn/split/add_patch.csv", index_col=0)
    label_data = TensorDataset(torch.Tensor(df[args.target]))
    data = StackDataset(data, label_data)
    split_dir = f"./split/add/{args.split}"

train_data = Subset(data, np.load(f"{split_dir}/train.npy"))
val_data = Subset(data, np.load(f"{split_dir}/val.npy"))

prefetch_factor = 5 if args.num_workers > 0 else None
persistent_workers = True if args.num_workers > 0 else False
loader = DataLoader(train_data, args.batch_size, True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
val_loader = DataLoader(val_data, args.batch_size*2, False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

model = Model()
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
if args.reg:
    dfscore = pd.DataFrame(columns=['epoch', 'R^2', 'RMSE', 'MAE'])
else:
    dfscore = pd.DataFrame(columns=['epoch', 'AUROC', 'AUPR'])
best_score = -math.inf
best_epoch = None
for epoch in range(args.n_epoch):
    
    model.train()
    for image_batch, label_batch in context(loader):
        optimizer.zero_grad()
        pred_batch = model(image_batch.to(device))
        if args.add:
            label_batch = label_batch[0]
        loss = criterion(pred_batch, label_batch.to(torch.float).to(device))
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    scheduler.step()
    epoch += 1   

    ## Save steps
    df = pd.DataFrame({'loss': losses})
    df.to_csv(f"{result_dir}/steps.csv", index_label='step')

    ## Save weight
    if epoch % 10 == 0 or epoch == args.n_epoch:
        torch.save(model.state_dict(), f"{result_dir}/models/{epoch}.pth")

    ## Evaluate
    logger.info("Evaluating...")
    model.eval()
    preds = []
    labels = []
    with torch.inference_mode():
        for i, (image_batch, label_batch) in enumerate(context(val_loader)):
            if args.add:
                label_batch = label_batch[0]
            pred_batch = model(image_batch.to(device))
            preds.append(pred_batch.cpu().numpy())
            labels.append(label_batch.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    if args.reg:
        dfscore.loc[epoch] = {
            'epoch': epoch, 
            'R^2': r2_score(labels, preds),
            'RMSE': mean_squared_error(labels, preds)**0.5,
            'MAE': mean_absolute_error(labels, preds)
        }
    else:
        labels = labels.astype(int)
        dfscore.loc[epoch] = {
            'epoch': epoch, 
            'AUROC': roc_auc_score(labels, preds),
            'AUPR': average_precision_score(labels, preds)
        }
    dfscore.to_csv(f"{result_dir}/score.csv", index=False)

    ## Early stopping
    if args.reg:
        score = -dfscore.loc[epoch, 'RMSE']
    else:
        score = dfscore.loc[epoch, 'AUROC']
    if best_score < score:
        if best_epoch is not None:
            os.remove(f"{result_dir}/models/{best_epoch}.pth")
        best_score = score
        best_epoch = epoch
        torch.save(model.state_dict(), f"{result_dir}/models/{epoch}.pth")
    else:
        if epoch - best_epoch >= args.early_stop:
            torch.save(model.state_dict(), f"{result_dir}/models/{epoch}.pth")
            break
print(f"training {args.studyname} finished!")