# -*- coding: utf-8 -*-
"""
# featurize module

@author: Katsuhisa MORITA
"""
import os
import re
import datetime
from typing import List, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import ImageOps, Image

import sslmodel
import sslmodel.sslutils as sslutils

# DataLoader
class Dataset_Batch(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                filein:str="",
                transform=None,
                ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # load data
        with open(filein, 'rb') as f:
            self.data = np.load(f)
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_data = Image.fromarray(out_data).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        return out_data

def prepare_dataset_batch(filein:str="", batch_size:int=32):
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
        filein=filein,
        transform=data_transform,
        )
    # to loader
    data_loader = sslmodel.data_handler.prep_dataloader(
        dataset, batch_size, 
        shuffle=False,
        drop_last=False)
    return data_loader

# Featurize Class
class Featurize:
    def __init__(self, DEVICE="cpu", lst_size=[], ):
        self.DEVICE=DEVICE
        self.lst_size=lst_size
        self.out_all=[np.zeros((0,size), dtype=np.float32) for size in self.lst_size]
        self.out_all_pool=[np.zeros((0,3*size), dtype=np.float32) for size in self.lst_size]
    
    def extraction():
        return None

    def featurize(self, model, data_loader, ):
        # featurize
        with torch.inference_mode():
            for data in data_loader:
                data = data.to(self.DEVICE)
                outs = self.extraction(model, data)
                for i, out in enumerate(outs):
                    self.out_all[i] = np.concatenate([self.out_all[i], out])
        
    def pooling(self, num_patch:int=200, ):
        for i, out in enumerate(self.out_all):
            self.out_all_pool[i] = np.concatenate([
                self.out_all_pool[i],
                self._pooling_array(out, num_patch=num_patch, size=self.lst_size[i])
                ])
        # reset output list
        self.out_all=[np.zeros((0,size), dtype=np.float32) for size in self.lst_size]

    def save_outpool(self, folder="", name=""):
        for i, out in enumerate(self.out_all_pool):
            np.save(f"{folder}/{name}_layer{i+1}.npy", out)

    def save_outall(self, folder="", name=""):
        for i, out in enumerate(self.out_all):
            np.save(f"{folder}/{name}_layer{i+1}.npy", out)

    def _pooling_array(self, out,num_patch=64, size=256):
        """ return max/min/mean pooling array with num_patch"""
        out = out.reshape(-1, num_patch, size)
        data_max=np.max(out,axis=1)
        data_min=np.min(out,axis=1)
        data_mean=np.mean(out,axis=1)
        data_all = np.concatenate([data_max, data_min, data_mean], axis=1).astype(np.float32)
        return data_all

class ResNet18Featurize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[64,64,128,256,512],
            )
    def extraction(self, model, x):
        x = model[0](x)# conv1
        x = model[1](x)# bn
        x = model[2](x)# relu
        x = model[3](x)# maxpool
        x1 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[4](x)# layer1
        x2 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[5](x)# layer2
        x3 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,128)
        x = model[6](x)# layer3
        x4 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,256)
        x = model[7](x)# layer4
        x5 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,512)
        return x1, x2, x3, x4, x5

class DenseNet121Featurize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[64,128,256,512,1024],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x = model[0][2](x)
        x = model[0][3](x)
        x1 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[0][4](x)
        x = model[0][5](x)
        x2 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,128)
        x = model[0][6](x)
        x = model[0][7](x)
        x3 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,256)
        x = model[0][8](x)
        x = model[0][9](x)
        x4 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,512)
        x = model[0][10](x)
        x = model[0][11](x)
        x5 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,1024)
        return x1, x2, x3, x4, x5

class EfficientNetB3Featurize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[24,32,48,136,1536],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x1 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,24)
        x = model[0][2](x)
        x2 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,32)
        x = model[0][3](x)
        x3 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,48)
        x = model[0][4](x)
        x = model[0][5](x)
        x4 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,136)
        x = model[0][6](x)
        x = model[0][7](x)
        x = model[0][8](x)
        x5 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,1536)
        return x1, x2, x3, x4, x5

class ConvNextTinyFeaturize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[96,192,384,768],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x1 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,96)
        x = model[0][2](x)
        x = model[0][3](x)
        x2 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,192)
        x = model[0][4](x)
        x = model[0][5](x)
        x3 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,384)
        x = model[0][6](x)
        x = model[0][7](x)
        x4 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,768)
        return x1, x2, x3, x4

class ConvNextTinyFeaturize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[96,192,384,768],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x1 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,96)
        x = model[0][2](x)
        x = model[0][3](x)
        x2 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,192)
        x = model[0][4](x)
        x = model[0][5](x)
        x3 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,384)
        x = model[0][6](x)
        x = model[0][7](x)
        x4 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,768)
        return x1, x2, x3, x4

class RegNetY16gfFeaturize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[32,48,120,336,888],
            )
    def extraction(self, model, x):
        x = model[0](x)
        x1 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,32)
        x = model[1][0](x)
        x2 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,48)
        x = model[1][1](x)
        x3 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,120)
        x = model[1][2](x)
        x4 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,336)
        x = model[1][3](x)
        x5 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,888)
        return x1, x2, x3, x4, x5

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

## Featurize Methods
# name: [Model_Class, last_layer_size, Featurize_Class]
DICT_MODEL = {
    "EfficientNetB3": [torchvision.models.efficientnet_b3, 1536, EfficientNetB3Featurize],
    "ConvNextTiny": [torchvision.models.convnext_tiny, 768, ConvNextTinyFeaturize],
    "ResNet18": [torchvision.models.resnet18, 512, ResNet18Featurize],
    "RegNetY16gf": [torchvision.models.regnet_y_1_6gf, 888, RegNetY16gfFeaturize],
    "DenseNet121": [torchvision.models.densenet121, 1024, DenseNet121Featurize],
}
DICT_SSL={
    "barlowtwins":sslutils.BarlowTwins,
    "swav":sslutils.SwaV,
    "byol":sslutils.Byol,
    "simsiam":sslutils.SimSiam,
    "wsl":sslutils.WSL,
}

def prepare_model(model_name:str='ResNet18', ssl_name="barlowtwins",  model_path="", pretrained=False, DEVICE="cpu"):
    """
    preparation of models
    Parameters
    ----------
        modelname (str)
            model architecture name

    """
    # model building with indicated name
    if pretrained:
        if model_name=="DenseNet121":
            encoder = DICT_MODEL[model_name][0](weights=None)
            encoder = _load_state_dict_dense(encoder, torch.load(model_path))
            model = nn.Sequential(
                *list(encoder.children())[:-1],
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
                )
        else:
            encoder = DICT_MODEL[model_name][0](weights=None)
            encoder.load_state_dict(torch.load(model_path))
            model=nn.Sequential(*list(encoder.children())[:-1])
    else:
        encoder = DICT_MODEL[model_name][0](weights=None)
        size = DICT_MODEL[model_name][1]
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
        ssl_class = DICT_SSL[ssl_name](DEVICE=DEVICE)
        model = ssl_class.prepare_featurize_model(
            backbone, model_path=model_path,
            head_size=size,
        )
    model.to(DEVICE)
    model.eval()
    return model

def featurize_layer(
    model, model_name="", ssl_name="",
    batch_size=128, lst_filein=list(), 
    folder_name="", result_name="", 
    DEVICE="cpu", num_patch=200,):
    try:
        extract_class = DICT_MODEL[model_name][2](DEVICE=DEVICE)
    except:
        print("indicated model name is not implemented")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # featurize
    for filein in lst_filein:
        data_loader=prepare_dataset_batch(filein=filein, batch_size=batch_size)
        extract_class.featurize(model, data_loader)
        extract_class.pooling(num_patch=num_patch)
    extract_class.save_outpool(folder=folder_name, name=result_name)