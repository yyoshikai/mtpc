# -*- coding: utf-8 -*-
"""
# weakly patch labeling

@author: Katsuhisa MORITA
"""
# path setting
PROJECT_PATH = '/work/gd43/a97001'

# packages installed in the current environment
import sys
import os
import gc
import datetime
import argparse
import time
import random

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# original packages in src
sys.path.append(f"{PROJECT_PATH}/src/SelfSupervisedLearningPathology")
import sslmodel
from sslmodel import data_handler as dh
import sslmodel.sslutils as sslutils

# argument
parser = argparse.ArgumentParser(description='CLI learning')
# base settings
parser.add_argument('--note', type=str, help='barlowtwins running')
parser.add_argument('--seed', type=int, default=0)
# data settings
parser.add_argument('--fold', type=int, default=0) # number of fold
parser.add_argument('--fold2', type=int, default=0) # number of fold2
parser.add_argument('--dir_result', type=str, help='result')
# model/learning settings
parser.add_argument('--model_name', type=str, default='ResNet18') # model architecture name
parser.add_argument('--num_epoch', type=int, default=50) # epoch
parser.add_argument('--batch_size', type=int, default=64) # batch size
parser.add_argument('--lr', type=float, default=0.01) # learning rate
parser.add_argument('--patience', type=int, default=3) # early stopping
parser.add_argument('--delta', type=float, default=0.002) # early stopping
parser.add_argument('--resume_epoch', type=int, default=20) # max repeat epoch for one run
parser.add_argument('--resize', action='store_true') # resize for test flag
parser.add_argument('--resume', action='store_true') # resume for test flag
# Transform (augmentation) settings
parser.add_argument('--color_plob', type=float, default=0.8)
parser.add_argument('--blur_plob', type=float, default=0.4)
parser.add_argument('--solar_plob', type=float, default=0.)

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

DICT_MODEL={
    "EfficientNetB3": [torchvision.models.efficientnet_b3, 1536],
    "ConvNextTiny": [torchvision.models.convnext_tiny, 768],
    "ResNet18": [torchvision.models.resnet18, 512],
    "RegNetY16gf": [torchvision.models.regnet_y_1_6gf, 888],
    "DenseNet121": [torchvision.models.densenet121, 1024],
}

# prepare data
class Dataset_Batch(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                fold:int=None,
                fold2:int=None,
                batch:int=None,
                transform=None,
                ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # load data
        if batch==3:
            self.data=np.concatenate([
                np.load(f"/work/gd43/share/tggates/liver/finding_fold/batch/fold{fold}_fold2{fold2}_batch3.npy"),
                np.load(f"/work/gd43/share/tggates/liver/finding_fold/batch/fold{fold}_fold2{fold2}_batch4.npy"),
            ],axis=0)
            self.label=np.concatenate([
                np.load(f"/work/gd43/share/tggates/liver/finding_fold/label/fold{fold}_fold2{fold2}_batch3.npy"),
                np.load(f"/work/gd43/share/tggates/liver/finding_fold/label/fold{fold}_fold2{fold2}_batch4.npy"),
            ],axis=0)
        else:
            self.data=np.load(f"/work/gd43/share/tggates/liver/finding_fold/batch/fold{fold}_fold2{fold2}_batch{batch}.npy")
            self.label=np.load(f"/work/gd43/share/tggates/liver/finding_fold/label/fold{fold}_fold2{fold2}_batch{batch}.npy")
        self.datanum = len(self.data)
        gc.collect()

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        # images
        out_data = self.data[idx]
        out_data = Image.fromarray(out_data).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        # labels
        out_label = self.label[idx]
        out_label = torch.Tensor(out_label)
        return out_data, out_label

def prepare_data(fold:int=None, fold2:int=None, batch:int=0, batch_size:int=32):
    """
    data preparation
    
    """
    # normalization
    train_transform = wsl.prepare_transform(
        color_plob=args.color_plob, blur_plob=args.blur_plob, solar_plob=args.solar_plob,
    )
    # data
    train_dataset = Dataset_Batch(
        fold=fold,
        fold2=fold2,
        batch=batch,
        transform=train_transform,
        )
    # resize for test
    if args.resize:
        train_dataset = dh.resize_dataset(train_dataset, size=128)
    # to loader
    train_loader = dh.prep_dataloader(train_dataset, batch_size)
    return train_loader

def prepare_valdata(fold:int=None, fold2:int=None, batch:int=0, batch_size:int=32):
    """
    data preparation
    
    """
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    # data
    dataset = Dataset_Batch(
        fold=fold,
        fold2=fold2,
        batch=batch,
        transform=data_transform,
        )
    # resize for test
    if args.resize:
        dataset = dh.resize_dataset(dataset, size=128)
    # to loader
    data_loader = dh.prep_dataloader(
        dataset, batch_size, 
        shuffle=False,
        drop_last=False
        )
    return data_loader

# model
def prepare_model(model_name:str='ResNet18', patience:int=7, delta:float=0, lr:float=0.003, num_epoch:int=150):
    """
    preparation of models
    Parameters
    ----------
        model_name (str)
            model architecture name
        
        patience (int)
            How long to wait after last time validation loss improved.

        delta (float)
            Minimum change in the monitored quantity to qualify as an improvement.

    """
    # model building with indicated name
    try:
        encoder = DICT_MODEL[model_name][0](weights=None)
        size=DICT_MODEL[model_name][1]
    except:
        print("indicated model name is not implemented")
        ValueError
    if model_name=="DenseNet121":
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
            )
    else:
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            )
    model, criterion = wsl.prepare_model(backbone, head_size=size, num_classes=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=0
        )
    early_stopping = sslmodel.utils.EarlyStopping(patience=patience, delta=delta, path=f'{DIR_NAME}/checkpoint.pt')
    return model, criterion, optimizer, scheduler, early_stopping

def load_model(model_name:str='ResNet18', patience:int=7, delta:float=0, lr:float=0.003, num_epoch:int=150):
    """
    preparation of models
    Parameters
    ----------
        model_name (str)
            model architecture name
        
        patience (int)
            How long to wait after last time validation loss improved.

        delta (float)
            Minimum change in the monitored quantity to qualify as an improvement.

    """
    # load
    state = torch.load(f'{DIR_NAME}/state.pt')
    # model building with indicated name
    try:
        encoder = DICT_MODEL[model_name][0](weights=None)
        size=DICT_MODEL[model_name][1]
    except:
        print("indicated model name is not implemented")
        ValueError
    if model_name=="DenseNet121":
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
            )
    else:
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            )
    model, _ = wsl.prepare_model(backbone, head_size=size, num_classes=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=0
        )
    epoch=state['epoch']
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])
    criterion = state['criterion']
    early_stopping = state['early_stopping']
    train_loss=state['train_loss']
    val_loss=state['val_loss']
    return model, criterion, optimizer, scheduler, early_stopping, train_loss, val_loss, epoch

# train epoch
def train_epoch(model, criterion, optimizer, epoch):
    """
    train for epoch
    with minibatch
    """
    model.train() # training
    train_epoch_loss = []
    val_epoch_loss = []
    # define loader
    lst_fold2=list(range(5))
    lst_fold2.remove(args.fold2)
    minibatch_lst=[[fold2,batch] for fold2 in lst_fold2 for batch in range(4)]
    # shuffle order of batch
    random.seed(args.seed+epoch)
    random.shuffle(minibatch_lst)
    random.seed(args.seed)
    # training
    model.train()
    for fold2, batch in minibatch_lst:
        # prep data
        train_loader=prepare_data(
            fold=args.fold,
            fold2=fold2,
            batch=batch, 
            batch_size=args.batch_size, 
            )
        # train
        for data, label in train_loader:
            loss = wsl.calc_loss(
                model, data, label, criterion,
            )
            train_epoch_loss.append(loss.item())
            optimizer.zero_grad() # reset gradients
            loss.backward() # backpropagation
            optimizer.step() # update parameters
        del train_loader
        gc.collect()
    # validation
    model.eval()
    with torch.inference_mode():
        for batch in range(4):
            val_loader=prepare_valdata(
                fold=args.fold,
                fold2=args.fold2,
                batch=batch,
                batch_size=args.batch_size,
            )
            for data, label in val_loader:
                loss = wsl.calc_loss(
                model, data, label, criterion,
                )
                val_epoch_loss.append(loss.item())
            del val_loader
            gc.collect()
    return model, np.mean(train_epoch_loss), np.mean(val_epoch_loss)

# train
def train(model, criterion, optimizer, scheduler, early_stopping, num_epoch:int=100, epoch_start:int=0, train_loss=list(), val_loss=list()):
    """ train ssl model """
    # settings
    start = time.time() # for time stamp
    for epoch in range(epoch_start, num_epoch):
        # train
        model, train_epoch_loss, val_epoch_loss = train_epoch(model, criterion, optimizer, epoch)
        scheduler.step()
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        LOGGER.logger.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}, val_loss: {val_epoch_loss:.4f}'
            )
        LOGGER.logger.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
        # save model
        state = {
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion":criterion,
            "early_stopping":early_stopping,
            "train_loss":train_loss,
            "val_loss":val_loss,
        }
        torch.save(state, f'{DIR_NAME}/state.pt')
        LOGGER.save_logger(fileout=file_log)
        # state check
        ## early stopping
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            LOGGER.logger.info(f'Early Stopping with Epoch: {epoch}')
            model.load_state_dict(torch.load(early_stopping.path))        
            return model, train_loss, val_loss, True
        ## time limit
        if epoch==epoch_start+args.resume_epoch-1:
            return None, None, None, False
    return model, train_loss, val_loss, True

def main(resume=False):
    # 1. Preparing
    if resume:
        model, criterion, optimizer, scheduler, early_stopping, train_loss, val_loss, epoch = load_model(
            model_name=args.model_name, patience=args.patience, delta=args.delta, lr=args.lr, num_epoch=args.num_epoch
        )
        epoch_start=epoch+1
    else:
        model, criterion, optimizer, scheduler, early_stopping = prepare_model(
            model_name=args.model_name, patience=args.patience, delta=args.delta, lr=args.lr, num_epoch=args.num_epoch
        )
        epoch_start=0
        train_loss=[]
        val_loss=[]
    # 2. Training
    model, train_loss, val_loss, flag_finish = train(
        model, criterion, optimizer, scheduler, early_stopping, num_epoch=args.num_epoch, epoch_start=epoch_start, train_loss=train_loss, val_loss=val_loss,
    )        
    # 3. save results & config
    if flag_finish:
        sslmodel.plot.plot_progress(train_loss, val_loss, DIR_NAME)
        sslmodel.utils.summarize_model(
            model,
            None,
            DIR_NAME, lst_name=['summary.txt', 'model.pt']
        )
        
        LOGGER.to_logger(name='argument', obj=args)
        LOGGER.to_logger(name='loss', obj=criterion)
        LOGGER.to_logger(
            name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
        )
        LOGGER.to_logger(name='scheduler', obj=scheduler)
    else:
        LOGGER.logger.info('reached max epoch / train')

if __name__ == '__main__':
    filename = os.path.basename(__file__).split('.')[0]
    DIR_NAME = PROJECT_PATH + '/result/' +args.dir_result # for output
    file_log = f'{DIR_NAME}/logger.pkl'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    # Set class
    wsl = sslutils.WSL(DEVICE=DEVICE)
    if args.resume:
        if not os.path.exists(file_log):
            print("log file doesnt exist")
        LOGGER = sslmodel.utils.logger_save()
        LOGGER.load_logger(filein=file_log)
        LOGGER.logger.info(f"resume: {datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
        main(resume=True)
    else:
        if not os.path.exists(DIR_NAME):
            os.makedirs(DIR_NAME)
        now = datetime.datetime.now().strftime('%H%M%S')
        LOGGER = sslmodel.utils.logger_save()
        LOGGER.init_logger(filename, DIR_NAME, now, level_console='debug') 
        main(resume=False)
