import sys, os
from argparse import ArgumentParser
from logging import getLogger
import torch
from torchvision import transforms as T
from glob import glob
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/mtpc"]
from src.featurize import featurize_mtpc
from src.model.backbone import get_backbone
from src.utils import set_random_seed
getLogger('tifffile').disabled = True

parser = ArgumentParser()
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--tqdm', action='store_true')
parser.add_argument('--seed', nargs='+', default=[0,1,2,3,4], type=int)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--size', type=int, default=1)
args = parser.parse_args()

backbone = get_backbone('resnet50')
transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]), 
])

for seed in args.seed:
    paths = sorted(glob(f"{WORKDIR}/mtpc/pretrain/results/250715_main/{seed}/**/resnet50.pth", recursive=True))
    paths = paths[args.rank::args.size]

    for path in paths:
        name = path.removeprefix(f"{WORKDIR}/mtpc/pretrain/results/") \
                .removesuffix("/resnet50.pth")

        out_dir = f"{WORKDIR}/mtpc/featurize/{name}"
        if os.path.exists(f"{out_dir}/feat_all.npy") and os.path.exists(f"{out_dir}/feat_added.npy"):
            print(f"Already finished: {name}")
            continue

        model_path = f"{WORKDIR}/mtpc/pretrain/results/{name}/resnet50.pth"
        if not os.path.exists(model_path):
            print(f"Model not found: {name}")
            continue

        print(f"Featurizing {name} ...")
        backbone.load_state_dict(torch.load(model_path, weights_only=True))
        set_random_seed(seed)
        featurize_mtpc(name, args.num_workers, args.batch_size, backbone, transform, args.tqdm)
