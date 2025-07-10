import sys, os
from argparse import ArgumentParser
from logging import getLogger
import torch
from torchvision import transforms as T
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/mtpc"]
from src.featurize import featurize_mtpc
from src.model.backbone import get_backbone
getLogger('tifffile').disabled = True

parser = ArgumentParser()
parser.add_argument('--name', help='... of pretrain/results/.../resnet50.pth')
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--tqdm', action='store_true')
args = parser.parse_args()

out_dir = f"{WORKDIR}/mtpc/featurize/{args.name}"
if os.path.exists(f"{out_dir}/feat_all.npy") and os.path.exists(f"{out_dir}/feat_added.npy"):
    print(f"Already finished: {args.name}")
    sys.exit()

model_path = f"{WORKDIR}/mtpc/pretrain/results/{args.name}/resnet50.pth"
if not os.path.exists(model_path):
    print(f"Model not found: {args.name}")
    sys.exit()

backbone = get_backbone('resnet50')
transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]), 
])
backbone.load_state_dict(torch.load(model_path, weights_only=True))
featurize_mtpc(args.name, args.num_workers, args.batch_size, backbone, transform, args.tqdm)
