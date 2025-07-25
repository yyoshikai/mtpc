# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 250612 Modified for moritasan's code.
from pathlib import Path
import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

from PIL import Image, ImageOps, ImageFilter
import numpy as np
from torch import nn, optim, amp
import torch
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/mtpc"]
from src.data import ApplyDataset
from src.pretrain import get_data
from src.utils import ddp_set_random_seed

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--mtpc-main', type=float, default=0.0)
parser.add_argument('--mtpc-add', type=float, default=0.0)
parser.add_argument('--tggate', type=float, default=0.0)
parser.add_argument('--mtpc-main-split', type=str, default=None)
parser.add_argument('--mtpc-add-split', type=str, default=None)
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')

# To moritasan
parser.add_argument('--z-normalization', choices=['bn', 'mean_std', 'mean_std_unbiased'], default='bn')
parser.add_argument('--optimizer', choices=['lars', 'adam'], default='lars')
parser.add_argument('--scheduler', choices=['cosine', 'warmup_cosine'], default='warmup_cosine')
parser.add_argument('--no-scaler', action='store_true')

parser.add_argument('--seed', type=int)

# continuous training
parser.add_argument('--init-weight')

def main():
    
    args = parser.parse_args()

    # If result exists, quit training
    if os.path.exists(args.checkpoint_dir / 'resnet50.pth'):
        print(f"{args.checkpoint_dir} has already finished.", flush=True)
        sys.exit()


    dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')

    # armだと発生する警告らしい。問題なさそうだが, 大量(workerごと)に表示されるので無視する。
    # 参考: https://github.com/pytorch/vision/issues/8574
    warnings.filterwarnings('ignore', "invalid value encountered in cast", RuntimeWarning)

    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    gpu = args.rank % torch.cuda.device_count()

    if args.seed is not None:
        ddp_set_random_seed(args.seed, torch.device('cuda', index=gpu))

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'w', buffering=1)
        steps_file = open(args.checkpoint_dir / 'steps.txt', 'w', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
        print("epoch,step,loss,on_diag,off_diag", file=steps_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    
    # load weight
    if args.init_weight is not None:
        model.backbone.load_state_dict(torch.load(args.init_weight, weights_only=True))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], bucket_cap_mb=100)

    if args.optimizer == 'lars':
        optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=True,
                        lars_adaptation_filter=True)
    else:
        optimizer = optim.Adam(parameters, lr=0)
    if args.scheduler == 'cosine':
        get_lr = get_lr_cosine
    elif args.scheduler == 'warmup_cosine':
        get_lr = get_lr_warmup_cosine

    start_epoch = 0

    # dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    dataset = get_data(args.mtpc_main, args.mtpc_add, args.tggate, args.seed, args.mtpc_main_split, args.mtpc_add_split)
    print(f"{len(dataset)=}")
    dataset = ApplyDataset(dataset, Transform())
    sampler = DistributedSampler(dataset, seed=args.seed or 0)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    if args.no_scaler:
        scaler = None
    else:
        scaler = amp.GradScaler('cuda')
    
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step, get_lr)
            optimizer.zero_grad()
            with amp.autocast('cuda'):
                loss, on_diag, off_diag, c = model.forward(y1, y2)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if step % args.print_freq == 0 or (step+1 == (epoch+1)*len(loader)):
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 on_diag=on_diag.item(),
                                 off_diag=off_diag.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
            if args.rank == 0:
                print(f"{epoch},{step},{loss.item()},{on_diag.item()},{off_diag.item()}", file=steps_file)

    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet50.pth')
        
    dist.destroy_process_group()

def get_lr_warmup_cosine(step, n_epoch, loader_size, batch_size):
    max_steps = n_epoch * loader_size
    warmup_steps = 10 * loader_size
    base_lr = batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    return lr

def get_lr_cosine(step, n_epoch, loader_size, batch_size):
    max_steps = n_epoch * loader_size

    base_lr = batch_size / 256
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * step / max_steps))
    return lr

def adjust_learning_rate(args, optimizer, loader, step, get_lr):
    
    lr = get_lr(step, args.epochs, len(loader), args.batch_size)   
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        if args.z_normalization == 'bn':
            self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        if self.args.z_normalization == 'bn':
            z1 = self.bn(z1)
            z2 = self.bn(z2)
        else:
            z1 = (z1-z1.mean(dim=0))/z1.std(dim=0, unbiased=self.args.z_normalization == 'mean_std_unbiased')
            z2 = (z2-z2.mean(dim=0))/z2.std(dim=0, unbiased=self.args.z_normalization == 'mean_std_unbiased')

        # empirical cross-correlation matrix
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        dist.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss, on_diag, off_diag, c


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


if __name__ == '__main__':
    main()
