# -*- coding: utf-8 -*-
"""
# test version (small scale)

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
from tqdm import tqdm
import sklearn.metrics as metrics

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
from sslmodel import utils
from sslmodel import plot
from sslmodel import data_handler as dh
import sslmodel.sslutils as sslutils
from sslmodel.models.linearhead import LinearHead

# argument
parser = argparse.ArgumentParser(description='CLI template')
# base settings
parser.add_argument('--note', type=str, help='note for this running')
parser.add_argument('--seed', type=int, default=24771)
# data settings
parser.add_argument('--dir_result', type=str, help='result')
# training settings
parser.add_argument('--model_name', type=str, default='ResNet18') # architecture name
parser.add_argument('--model_path', type=str, default='file.pth') # if pretrained/trained model used
parser.add_argument('--pretrained', action='store_true') # pretarined model
parser.add_argument('--load_model', action='store_true') # trained model
# ssl settings
parser.add_argument('--ssl_name', type=str, default='barlowtwins') # ssl architecture name
parser.add_argument('--num_epoch_ssl', type=int, default=50) # epoch
parser.add_argument('--batch_size_ssl', type=int, default=64) # batch size ssl
parser.add_argument('--lr_ssl', type=float, default=0.003) # learning rate for ssl
parser.add_argument('--patience_ssl', type=int, default=5) # early stopping for ssl
parser.add_argument('--delta_ssl', type=float, default=0.0) # early stopping for ssl
# linear
parser.add_argument('--num_epoch', type=int, default=150) # epoch
parser.add_argument('--batch_size', type=int, default=128) # batch size
parser.add_argument('--lr', type=float, default=0.0003) # learning rate
parser.add_argument('--patience', type=int, default=5) # early stopping
parser.add_argument('--delta', type=float, default=0.) # early stopping
# Transform (augmentation) settings
parser.add_argument('--color_plob', type=float, default=0.8)
parser.add_argument('--blur_plob', type=float, default=0.4)
parser.add_argument('--solar_plob', type=float, default=0.)

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

DICT_MODEL={
    "EfficientNetB3": [torchvision.models.efficientnet_b3, 1536],
    "ConvNextTiny": [torchvision.models.convnext_tiny, 768],
    "ResNet50": [torchvision.models.resnet50, 2048],
    "ResNet18": [torchvision.models.resnet18, 512],
    "RegNetY16gf": [torchvision.models.regnet_y_1_6gf, 888],
    "DenseNet121": [torchvision.models.densenet121, 1024],
}
DICT_SSL={
    "barlowtwins":sslutils.BarlowTwins,
    "swav":sslutils.SwaV,
    "byol":sslutils.Byol,
    "simsiam":sslutils.SimSiam,
}

# prepare data
class ColonDataset(torch.utils.data.Dataset):
    """ to create my dataset """
    def __init__(self, 
                split:str='train',
                transform=None):
        if type(transform)!=list:
            self.transform = [transform]
        else:
            self.transform = transform

        # load from project folder
        DATA = np.load('/work/gd43/share/colon224.npz')
        self.data = DATA[f'{split}_images']
        self.label = DATA[f'{split}_labels']
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_label = self.label[idx].astype(int)
        out_data = Image.fromarray(out_data).convert("RGB")
        if self.transform:
            for t in self.transform:
                out_data = t(out_data)
        return out_data, out_label

def prepare_ssl_data(batch_size:int=32, ):
    """
    data preparation
    
    """
    # normalization
    train_ssl_transform = ssl_class.prepare_transform(
        color_plob=args.color_plob, blur_plob=args.blur_plob, solar_plob=args.solar_plob,
    )
    # data
    train_ssl_dataset = ColonDataset(
        split="train",
        transform=train_ssl_transform,
        )
    # to loader
    train_ssl_loader = dh.prep_dataloader(train_ssl_dataset, batch_size, shuffle=True, drop_last=True)
    return train_ssl_loader

def prepare_data(batch_size:int=32, ):
    """
    data preparation
    
    """
    train_transform = utils.ssl_transform(
        split=False, size=(224,224),
        color_plob=args.color_plob, blur_plob=args.blur_plob, solar_plob=args.solar_plob,
    )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    train_dataset = ColonDataset(
        split="train",
        transform=train_transform,
        )
    val_dataset = ColonDataset(
        split="val",
        transform=test_transform,
        )
    test_dataset = ColonDataset(
        split="test",
        transform=test_transform,
        )
    # to loader
    train_loader = dh.prep_dataloader(train_dataset, batch_size, shuffle=True, drop_last=True)
    val_loader = dh.prep_dataloader(val_dataset, batch_size, shuffle=False, drop_last=False)
    test_loader = dh.prep_dataloader(test_dataset, batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

def _load_state_dict_dense(model, weights):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls (pretrained-models). This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    for key in list(weights.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            weights[new_key] = weights[key]
            del weights[key]
    model.load_state_dict(weights)
    return model

def prepare_model(model, model_name="", patience:int=7, delta:float=0, lr:float=0.003, num_epoch:int=150,  num_classes:int=9, pretrained=False,):
    """
    preparation of models
    Parameters
    ----------
        model
            self-supervised learned model

        patience (int)
            How long to wait after last time validation loss improved.

        delta (float)
            Minimum change in the monitored quantity to qualify as an improvement.

    """
    # model building
    if pretrained:
        if model_name=="DenseNet121":
            encoder = DICT_MODEL[model_name][0](weights=None)
            encoder = _load_state_dict_dense(encoder, torch.load(args.model_path))
            model = nn.Sequential(
                *list(encoder.children())[:-1],
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
                )
        else:
            encoder = DICT_MODEL[model_name][0](weights=None)
            encoder.load_state_dict(torch.load(args.model_path))
            model=nn.Sequential(*list(encoder.children())[:-1])
        model_all = LinearHead(
            model, 
            num_classes=num_classes, 
            dim=DICT_MODEL[model_name][1]) 
    else:
        if args.ssl_name=="simsiam" :
            backbone=model.encoder
        elif args.ssl_name=="barlowtwins" or args.ssl_name=="swav":
            backbone=model.backbone
        elif args.ssl_name=="byol":
            backbone=model.net
        model_all = LinearHead(
            backbone, 
            num_classes=num_classes, 
            dim=DICT_MODEL[model_name][1]) 
    model_all = utils.fix_params(model_all, forall=False)
    model_all.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_all.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=0
        )
    early_stopping = utils.EarlyStopping(patience=patience, delta=delta, path=f'{DIR_NAME}/checkpoint.pt')
    return model_all, criterion, optimizer, scheduler, early_stopping

def prepare_ssl_model(model_name:str='ResNet18', patience:int=7, delta:float=0, lr:float=0.003, num_epoch:int=150):
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
    model, criterion = ssl_class.prepare_model(backbone, head_size=size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=0
        )
    early_stopping = sslmodel.utils.EarlyStopping(patience=patience, delta=delta, path=f'{DIR_NAME}/checkpoint.pt')
    return model, criterion, optimizer, scheduler, early_stopping

# train epoch
def train_epoch_ssl(model, train_loader, criterion, optimizer):
    """
    train for epoch
    
    """
    model.train() # training
    train_batch_loss = []
    for data, _ in train_loader:
        loss = ssl_class.calc_loss(
            model, data, criterion,
        )
        train_batch_loss.append(loss.item())
        optimizer.zero_grad() # reset gradients
        loss.backward() # backpropagation
        optimizer.step() # update parameters
    return model, np.mean(train_batch_loss)

def train_epoch(model, train_loader, val_loader, criterion, optimizer):
    """
    train for epoch
    
    """
    model.train() # training
    train_batch_loss = []
    for data, label in train_loader:
        data, label = data.to(DEVICE), label.to(DEVICE) # put data on GPU
        optimizer.zero_grad() # reset gradients
        output = model(data) # forward
        loss = criterion(output, label.squeeze_()) # calculate loss
        loss.backward() # backpropagation
        optimizer.step() # update parameters
        train_batch_loss.append(loss.item())
    model.eval() # test (validation)
    val_batch_loss = []
    with torch.inference_mode():
        for data, label in val_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            loss = criterion(output, label.squeeze_())
            val_batch_loss.append(loss.item())
    return model, np.mean(train_batch_loss), np.mean(val_batch_loss)

# train
def train_ssl(model, train_loader, criterion, optimizer, scheduler, early_stopping, num_epoch:int=100):
    """ train ssl model """
    start = time.time() # for time stamp
    train_loss=[]
    for epoch in range(num_epoch):
        model, train_epoch_loss = train_epoch_ssl(model, train_loader, criterion, optimizer)
        scheduler.step()
        train_loss.append(train_epoch_loss)
        LOGGER.logger.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}'
            )
        LOGGER.logger.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
        early_stopping(train_epoch_loss, model)
        if early_stopping.early_stop:
            LOGGER.logger.info(f'Early Stopping with Epoch: {epoch}')
            model.load_state_dict(torch.load(early_stopping.path))        
            return model, train_loss
    return model, train_loss

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epoch:int=100):
    """ train main model """
    train_loss = []
    val_loss = []
    for epoch in range(num_epoch):
        model, train_epoch_loss, val_epoch_loss = train_epoch(
            model, train_loader, val_loader, criterion, optimizer
            )
        scheduler.step() # should be removed if not necessary
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        LOGGER.logger.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}, val_loss: {val_epoch_loss:.4f}'
            )
        early_stopping(val_epoch_loss, model) # early stopping
        if early_stopping.early_stop:
            LOGGER.logger.info(f'Early Stopping with Epoch: {epoch}')
            model.load_state_dict(torch.load(early_stopping.path))        
            break
    return model, train_loss, val_loss

# predict
def predict(model, dataloader):
    """prediction"""
    model.eval()
    y_true = torch.tensor([]).to(DEVICE)
    y_pred = torch.tensor([]).to(DEVICE)
    with torch.inference_mode():
        for data, label in dataloader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            output = output.softmax(dim=-1) # pay attention: softmax function
            y_true = torch.cat((y_true, label), 0)
            y_pred = torch.cat((y_pred, output), 0)
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        df_res, acc, ba, auroc=evaluate(y_true.flatten().astype(int), y_pred)
    return df_res, acc, ba, auroc

def evaluate(y_true, y_pred):
    """
    scoring module
    Parameters
    ----------
        y_true
            the ground truth labels, shape: (n_samples,)
        y_pred
            the predicted score of each class, shape: (n_samples, n_classes)

    """
    # Macro Indicators
    lst_res=[]
    for i in range(max(y_true)+1):
        try:
            auroc = metrics.roc_auc_score(y_true == i, y_pred[:, i])
            precision, recall, thresholds = metrics.precision_recall_curve(y_true == i, y_pred[:, i])
            aupr = metrics.auc(recall, precision)
            mAP = metrics.average_precision_score(y_true == i, y_pred[:, i])
        except:
            auroc, auppr, mAP = np.nan, np.nan, np.nan
        ba=metrics.balanced_accuracy_score(y_true == i, [np.rint(v) for v in y_pred[:, i]])
        lst_res.append([auroc, aupr, mAP, ba])
    df_res=pd.DataFrame(lst_res, columns=["AUROC","AUPR","mAP","Balanced Accuracy"]).T
    df_res["Macro Average"]=df_res.mean(axis=1)
    # Micro Indicators
    acc = np.mean(np.argmax(y_pred, axis=1) == y_true)
    ba = metrics.balanced_accuracy_score(y_true, np.argmax(y_pred, axis=1))
    auroc = metrics.roc_auc_score(y_true, y_pred, average="micro", multi_class="ovr")
    return df_res, acc, ba, auroc

def main():
    # 1. Self-Supervised Training
    if args.load_model:
        model = prepare_ssl_model(
            model_name=args.model_name, 
            lr=args.lr_ssl, num_epoch=args.num_epoch_ssl,
            patience=args.patience_ssl, delta=args.delta_ssl, 
        )[0]
        model.load_state_dict(torch.load(args.model_path))
    elif not args.pretrained:
        train_ssl_loader = prepare_ssl_data(batch_size=args.batch_size_ssl)
        model, criterion, optimizer, scheduler, early_stopping = prepare_ssl_model(
            model_name=args.model_name, 
            lr=args.lr_ssl, num_epoch=args.num_epoch_ssl,
            patience=args.patience_ssl, delta=args.delta_ssl, 
        )
        model, train_loss, = train_ssl(
            model, train_ssl_loader, 
            criterion, optimizer, scheduler, early_stopping, 
            num_epoch=args.num_epoch_ssl,
        )        
        plot.plot_progress_train(train_loss, DIR_NAME)
        sslmodel.utils.summarize_model(
            model, None,
            DIR_NAME, lst_name=['summary_ssl.txt', 'model_ssl.pt']
        )
    else:
        model=None
    # 2. Classifier Training
    train_loader, val_loader, test_loader = prepare_data(batch_size=args.batch_size)
    model, criterion, optimizer, scheduler, early_stopping = prepare_model(
        model, model_name=args.model_name,
        lr=args.lr, num_epoch=args.num_epoch,
        patience=args.patience, delta=args.delta, pretrained=args.pretrained)
    model, train_loss, val_loss = train(
        model, train_loader, val_loader, 
        criterion, optimizer, scheduler, early_stopping, 
        num_epoch=args.num_epoch,
        )
    plot.plot_progress(train_loss, val_loss, DIR_NAME)
    sslmodel.utils.summarize_model(
        model, None,
        DIR_NAME, lst_name=['summary.txt', 'model.pt']
    )
    # 3. evaluation
    res1 = predict(model, train_loader)
    LOGGER.logger.info(f'train acc: {res1[1]:.4f}, auc{res1[0].loc["AUROC","Macro Average"]: 4f}')
    res2 = predict(model, val_loader)
    LOGGER.logger.info(f'val acc: {res2[1]:.4f}, auc{res2[0].loc["AUROC","Macro Average"]: 4f}')
    res3 = predict(model, test_loader)
    LOGGER.logger.info(f'test acc: {res3[1]:.4f}, auc{res3[0].loc["AUROC","Macro Average"]: 4f}')
    pd.to_pickle([res1, res2, res3], f"{DIR_NAME}/result.pickle")

if __name__ == '__main__':
    filename = os.path.basename(__file__).split('.')[0]
    DIR_NAME = PROJECT_PATH + '/result/' +args.dir_result # for output
    file_log = f'{DIR_NAME}/logger.pkl'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    # Set SSL class
    ssl_class=DICT_SSL[args.ssl_name](DEVICE=DEVICE)
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    now = datetime.datetime.now().strftime('%H%M%S')
    LOGGER = sslmodel.utils.logger_save()
    LOGGER.init_logger(filename, DIR_NAME, now, level_console='debug') 
    main()
