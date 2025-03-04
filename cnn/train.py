import sys, os, argparse, yaml
from tqdm import tqdm
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import StackDataset, Subset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score, average_precision_score
WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path += [f'{WORKDIR}/mtpc', WORKDIR]
from src.utils.logger import add_stream_handler, add_file_handler, get_logger
from src.utils.path import make_dir
from src.data import untuple_dataset, MTPCDataset, BaseAugmentDataset, InDataset
LABEL_TYPES = ['Other', 'Normal', 'Mild', 'Moderate', 'Severe']

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--studyname", required=True)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--split", required=True)
parser.add_argument("--n-epoch", type=int, default=20)
parser.add_argument("--positive-th", required=True, choices=LABEL_TYPES[1:])
parser.add_argument("--num-workers", type=int, default=28)
parser.add_argument("--tqdm", action='store_true')
parser.add_argument("--compile", action='store_true')
parser.add_argument("--duplicate", default='ask')
parser.add_argument("--early-stop", type=int)
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
data = MTPCDataset(256)
image_data, label_data = untuple_dataset(data, 2)
image_data = BaseAugmentDataset(image_data)
label_data = InDataset(label_data, LABEL_TYPES[LABEL_TYPES.index(args.positive_th):])
data = StackDataset(image_data, label_data)
train_data = Subset(data, np.load(f"./split/results/{args.split}/train.npy"))
test_data = Subset(data, np.load(f"./split/results/{args.split}/test.npy"))

loader = DataLoader(train_data, args.batch_size, True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=5, persistent_workers=True)
test_loader = DataLoader(test_data, args.batch_size*2, False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=5, persistent_workers=True)

# Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(2048, 128),
            nn.GELU(),
            nn.Linear(128, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x.squeeze_(-1, -2)
        x = self.head(x)
        x.squeeze_(-1)
        return x
model = Model()
model.to(device)
if args.compile:
    model = torch.compile(model)
criterion = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters())
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Train
context = tqdm if args.tqdm else lambda x: x
losses = []
dfscore = pd.DataFrame(columns=['epoch', 'AUROC', 'AUPR'])
best_score = 0.0
stop_epoch = 0
for epoch in range(args.n_epoch):
    
    model.train()
    for image_batch, label_batch in context(loader):
        optimizer.zero_grad()
        pred_batch = model(image_batch.to(device))
        loss = criterion(pred_batch, label_batch.to(device).to(torch.float))
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
        for i, (image_batch, label_batch) in enumerate(context(test_loader)):
            pred_batch = model(image_batch.to(device))
            preds.append(pred_batch.cpu().numpy())
            labels.append(label_batch.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    dfscore.loc[epoch] = {
        'epoch': epoch, 
        'AUROC': roc_auc_score(labels, preds),
        'AUPR': average_precision_score(labels, preds)
    }
    dfscore.to_csv(f"{result_dir}/score.csv", index=False)

    ## Early stopping
    score = dfscore.loc[epoch, 'AUROC']
    if best_score < score:
        best_score = score
        stop_epoch = 0
    else:
        stop_epoch += 1
        if stop_epoch >= 5:
            torch.save(model.state_dict(), f"{result_dir}/models/{epoch}.pth")
            break
