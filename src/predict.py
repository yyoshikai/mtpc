import sys, os, math
import itertools as itr
from glob import glob
from typing import Optional
from logging import getLogger
import numpy as np, pandas as pd
from tqdm import tqdm as _tqdm
from schedulefree import RAdamScheduleFree
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from .utils.logger import get_logger

def get_mask(y: np.ndarray, folds: np.ndarray, reg: bool) \
        -> tuple[np.ndarray|None]:
    logger = get_logger(__name__)

    test_mask = folds == 0
    if reg:
        val_fold = 1
    else:
        y_test = y[test_mask]
        if np.all(y_test == y_test[0]):
            logger.info(f"y_test only contains {y_test[0]}")
            return None, None, None
        for val_fold in range(1, 5):
            val_mask = folds == val_fold
            train_mask = (folds > 0)&(folds != val_fold)
            y_val = y[val_mask]
            y_train = y[train_mask]
            if not (np.all(y_train == y_train[0]) or np.all(y_val == y_val[0])):
                break
        else:
            logger.info(f"No val_fold could split property.")
            return None, None, None
    val_mask = folds == val_fold
    train_mask = (folds > 0)&(folds != val_fold)
    assert not np.any(train_mask&val_mask)
    assert not np.any(val_mask&test_mask)
    assert not np.any(test_mask&train_mask)
    return train_mask, val_mask, test_mask

def predict(model: nn.Module,
    reg: bool, 
    result_dir: str,
    n_epoch: int,
    early_stop: int,
    lr: float,
    output_std: float,
    train_loader: DataLoader,
    test_loader: DataLoader, 
    val_loader: Optional[DataLoader] = None,
    optimizer: str = 'adam',

    compile: bool=False, 
    tqdm: bool=False,
    save_steps: bool=False,
    save_pred: bool=False, 
    save_model: bool=False,

):

    # Result exists?
    result_names = ['score.csv', 'train_score.csv']
    if val_loader is not None:
        result_names += ['val_score.csv', 'train_val_score.csv']
    if save_steps: result_names.append('steps.csv')
    if save_model: result_names.append('best_model_*.pth')
    if save_pred:
        result_names += ['preds.csv', 'train_preds.csv']
        if val_loader is not None:
            result_names += ['val_preds.csv', 'train_val_preds.csv']
    if all(len(glob(f"{result_dir}/{name}")) > 0 for name in result_names):
        print(f"All results exist for {result_dir}")
        return

    # Environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = getLogger("predict")
    
    # Model
    model.to(device)
    if compile:
        model = torch.compile(model)
    if reg:
        criterion = nn.MSELoss(reduction='mean')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    match optimizer:
        case 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            optimizer_mode = False
        case 'radam_free':
            optimizer = RAdamScheduleFree(model.parameters(), lr=lr)
            scheduler = None
            optimizer_mode = True
            import torch.nn.functional as F
            # bn採用時は何かしないといけないらしいので確認(F.batch_normを使っていたら把握できないが。。)。
            for mod in model.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    raise ValueError(f"RAdamScheduleFree not supported for module with BatchNorm")
        case _:
            raise NotImplementedError


    # Train
    context = _tqdm if tqdm else lambda x: x
    losses = []
    if val_loader is not None:
        with open(f"{result_dir}/val_scores.csv", 'w') as f:
            if reg:
                f.write("epoch,R^2,RMSE,MAE,rho,p_rho\n")
            else:
                f.write("epoch,AUROC,AUPR")

    best_score = -math.inf
    best_epoch = None
    for epoch in range(n_epoch):
        
        model.train()
        if optimizer_mode: optimizer.train()
        for input_batch, target_batch in context(train_loader):
            optimizer.zero_grad()
            pred_batch = model(input_batch.to(device))
            loss = criterion(pred_batch, target_batch.to(torch.float).to(device)) / output_std
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        epoch += 1   

        ## Save steps
        if save_steps:
            df = pd.DataFrame({'loss': losses})
            df.to_csv(f"{result_dir}/steps.csv", index_label='step')

        model.eval()
        if optimizer_mode: optimizer.eval()
        
        ## Evaluate
        if val_loader is not None:
            logger.info(f"Evaluating epoch {epoch}...")
            preds = []
            targets = []
            with torch.inference_mode():
                for input_batch, target_batch in context(val_loader):
                    pred_batch = model(input_batch.to(device))
                    preds.append(pred_batch.cpu().numpy())
                    targets.append(target_batch.numpy())
            preds = np.concatenate(preds)
            targets = np.concatenate(targets)
            with open(f"{result_dir}/val_scores.csv", 'a') as f:
                if reg:
                    rho = spearmanr(targets, preds)
                    score = -mean_squared_error(targets, preds)
                    f.write(f"{epoch},{r2_score(targets,preds)},{mean_squared_error(targets, preds)**0.5},{mean_absolute_error(targets, preds)},{rho.statistic},{rho.pvalue}\n")
                else:
                    targets = targets.astype(int)
                    f.write(f"{epoch},{roc_auc_score(targets, preds)},{average_precision_score(targets, preds)}\n")
                    score = roc_auc_score(targets, preds)

            ## Early stopping
            if best_score < score:
                if best_epoch is not None:
                    os.remove(f"{result_dir}/best_model_{best_epoch}.pth")
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), f"{result_dir}/best_model_{epoch}.pth")
            else:
                if epoch - best_epoch >= early_stop:
                    break
        else:
            if epoch == n_epoch:
                torch.save(model.state_dict(), f"{result_dir}/best_model_{epoch}.pth")
                best_epoch = epoch

    # Evaluate for test data
    logger.info(f"Evaluating with best model ({best_epoch})...")
    if val_loader is not None:
        model.load_state_dict(torch.load(f"{result_dir}/best_model_{best_epoch}.pth", weights_only=True))
    model.eval()

    train_preds = train_targets = val_preds = val_targets = None
    for loader, split in zip([train_loader, val_loader, None, test_loader], ['train_', 'val_', 'train_val_', '']):

        if split == 'train_val_':
            if val_preds is None: continue
            targets = np.concatenate([train_targets, val_targets])
            preds = np.concatenate([train_preds, val_preds])
        else:
            if loader is None: continue
            preds = []
            targets = []
            with torch.inference_mode():
                for input, target in context(loader):
                    pred = model(input.to(device))
                    preds.append(pred.cpu().numpy())
                    targets.append(target.numpy())
            preds = np.concatenate(preds)
            targets = np.concatenate(targets)
            if split == 'train_':
                train_preds, train_targets = preds, targets
            elif split == 'val_':
                val_preds, val_targets = preds, targets
        
        if save_pred:
            pd.DataFrame({'pred': preds, 'target': targets}).to_csv(f"{result_dir}/{split}preds.csv", index=False)
        if reg:
            res = spearmanr(preds, targets)
            df = pd.DataFrame({'score': {
                'RMSE': mean_squared_error(targets, preds)**0.5,
                'MAE': mean_absolute_error(targets, preds),
                'R^2': r2_score(targets, preds),
                'rho': res.statistic,
                'p_rho': res.pvalue
            }})
        else:
            targets = targets.astype(int)
            df = pd.DataFrame({'score': {
                'AUROC': roc_auc_score(targets, preds),
                'AUPR': average_precision_score(targets, preds),
            }})
        df.to_csv(f"{result_dir}/{split}score.csv")

    # Remove best model
    if not save_model:
        os.remove(f"{result_dir}/best_model_{best_epoch}.pth")
    logger.info(f"training {result_dir} finished!")

