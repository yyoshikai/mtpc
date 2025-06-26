import sys, os
from argparse import ArgumentParser
import torch
from torchvision import transforms

sys.path += ["/workspace/mtpc", "/workspace/mtpc/pretrain/vicreg"]

from src.model.backbone import get_backbone
from pretrain.vicreg.main_vicreg_tggate import get_arguments
from src.featurize import featurize_mtpc
from src.utils.logger import get_logger, add_stream_handler


parser = ArgumentParser()
parser.add_argument('--sname')
args = parser.parse_args()

backbone = get_backbone('resnet50')

# /workspace/mtpc/pretrain/vicreg/evaluate.py より
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
])

rdir = f"/workspace/mtpc/pretrain/vicreg/results/{args.sname}"
weight_path = f"{rdir}/resnet50.pth"
if not os.path.exists(weight_path):
    print(f"{weight_path} does not exist.")
    sys.exit()

logger = get_logger()
add_stream_handler(logger)


stat_path = f"{rdir}/stats.txt"
if os.path.exists(stat_path):
    with open(stat_path, 'r') as f:
        argv = f.readline()[:-1].split(' ')
        train_parser = get_arguments()
        args, _ = train_parser.parse_known_args(argv[1:])
        epoch  = args.epochs
else:
    epoch = 'unk'

weight = torch.load(weight_path, weights_only=True)
backbone.load_state_dict(weight)

featurize_mtpc(f"vicreg/{args.sname}/{epoch}", 28, 512, backbone, transform)
