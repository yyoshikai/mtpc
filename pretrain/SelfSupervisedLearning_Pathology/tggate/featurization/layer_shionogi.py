# -*- coding: utf-8 -*-
"""
# predict / pooling

@author: Katsuhisa MORITA
"""
# path setting
PROJECT_PATH = '/work/gd43/a97001'

# packages installed in the current environment
import sys
import datetime
import argparse
import time

import numpy as np
import pandas as pd
import torch

# original packages in src
sys.path.append(f"{PROJECT_PATH}/src/SelfSupervisedLearningPathology")
from tggate import featurize
import sslmodel

# argument
parser = argparse.ArgumentParser(description='CLI inference')
parser.add_argument('--note', type=str, help='feature')
parser.add_argument('--seed', type=int, default=24771)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model_name', type=str, default='ResNet18') # model architecture name
parser.add_argument('--ssl_name', type=str, default='barlowtwins') # ssl architecture name
parser.add_argument('--dir_model', type=str, default='')
parser.add_argument('--result_name', type=str, default='')
parser.add_argument('--folder_name', type=str, default='')
parser.add_argument('--pretrained', action='store_true')

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

def main():
    # settings
    start = time.time() # for time stamp
    print(f"start: {start}")
    # 1. model data preparation
    # model
    model = featurize.prepare_model(
        model_name=args.model_name, 
        ssl_name=args.ssl_name,
        model_path=args.dir_model,
        pretrained=args.pretrained, DEVICE=DEVICE
        )
    # load file names
    df_info=pd.read_csv(f"{PROJECT_PATH}/experiments_pharm/shionogi_info.csv")
    df_info=df_info.sort_values(by=["INDEX"])
    lst_filein=df_info["FILE"].tolist()
    # 2. inference & save results
    featurize.featurize_layer(
        model, model_name=args.model_name, ssl_name=args.ssl_name,
        batch_size=args.batch_size, lst_filein=lst_filein,
        folder_name=args.folder_name, result_name=args.result_name,
        DEVICE=DEVICE, num_patch=NUM_PATCH)
    # 3. save config
    print('elapsed_time: {:.2f} min'.format((time.time() - start)/60))        

if __name__ == '__main__':
    NUM_PATCH=200
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    main()