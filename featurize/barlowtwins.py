import sys
from argparse import ArgumentParser
import torch
from torchvision import transforms

sys.path.append("/workspace/mtpc")
from src.model.backbone import get_backbone
from src.model.state_dict import state_modifiers
from src.featurize import featurize_mtpc

parser = ArgumentParser()
parser.add_argument('--sname')
parser.add_argument('--epoch', default='unk')
parser.add_argument('--weight', choices=['model', 'checkpoint_latest', 'checkpoint_best'], 
        default='model')
args = parser.parse_args()

backbone = get_backbone('resnet50')

# /workspace/mtpc/pretrain/barlowtwins/evaluate.py より
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
])

rdir = f"/workspace/mtpc/pretrain/barlowtwins/results/{args.sname}"
if args.weight == 'model':
    weight = torch.load(f"{rdir}/resnet50.pth", weights_only=True)
else:
    name = args.weight.split('_')[1] 
    checkpoint = torch.load(F"{rdir}/checkpoints/{name}.pth")
    weight = state_modifiers['barlowtwins'](checkpoint)
    assert args.epoch == 'unk'
    args.epoch = checkpoint['epoch']

backbone.load_state_dict(weight)
featurize_mtpc(f"barlowtwins/{args.sname}/{args.epoch}", 28, 512, backbone, transform)
