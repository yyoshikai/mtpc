import sys, os, yaml
from argparse import ArgumentParser
import torch, torch.nn as nn
from addict import Dict
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/mtpc"]
from src.model.backbone import get_backbone
from src.utils.model import get_substate
from src.model.barlowtwins import BarlowTwins
from featurize.featurize import featurize_mtpc

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--sname', required=True)
    parser.add_argument('--epoch', type=int, default=29)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--remove-last-relu', action='store_true')
    parser.add_argument('--from-imagenet')
    parser.add_argument('--bsz', type=int)
    args = parser.parse_args()

    rdir = f"/workspace/mtpc/pretrain/bt/results/{args.sname}"
    
    with open(f"{rdir}/args.yaml") as f:
        pargs = Dict(yaml.safe_load(f))
    batch_size = args.bsz or pargs['bsz']

    snameh = args.sname
    if args.remove_last_relu: snameh += '_norelu'

    structure = pargs.get('structure', 'resnet18')
    scheme = pargs.get('scheme', 'bt')

    backbone = get_backbone(structure, weight=None)

    
    
    backbone.load_state_dict(get_substate(torch.load(f"{rdir}/models/{args.epoch}.pth", weights_only=True), 'backbone.'))
    if args.remove_last_relu:
        backbone[-2][-1].relu2 = nn.Identity()

    model = BarlowTwins.from_args(backbone, pargs)    
    transform = model.get_eval_transform()
    featurize_mtpc(f"resnet18/{scheme}/{snameh}/{args.epoch}", args.num_workers, args.bsz, backbone, transform)
