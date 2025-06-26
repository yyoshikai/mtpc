import sys, os
from argparse import ArgumentParser
import torch
from torchvision import transforms

sys.path += ["/workspace/mtpc", "/workspace/mtpc/pretrain/VICRegL"]

from src.model.backbone import get_backbone
from src.model.state_dict import state_modifiers
from src.featurize import featurize_mtpc


parser = ArgumentParser()
parser.add_argument('--sname')
args = parser.parse_args()

backbone = get_backbone('resnet50')

# /workspace/mtpc/pretrain/VICRegL/evaluate.py より
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
])

rdir = f"/workspace/mtpc/pretrain/VICRegL/results/{args.sname}"
weight_path = f"{rdir}/model.pth"
if not os.path.exists(weight_path):
    print(f"{weight_path} does not exist.")
    sys.exit()
checkpoint = torch.load(weight_path)

epoch = checkpoint['epoch']
assert epoch == 30
backbone.load_state_dict(state_modifiers['VICRegL'](checkpoint))

featurize_mtpc(f"VICRegL/{args.sname}/{epoch}", 28, 512, backbone, transform)
