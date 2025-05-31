import sys, os, argparse
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/mtpc"]
from src.model.backbone import structures, get_backbone, structure2weights
from src.model.barlowtwins import BarlowTwins
from src.pretrain import pretrain

DDIR = f"{WORKDIR}/cheminfodata/mtpc"

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--studyname", default='default')
parser.add_argument('--data', nargs='+', default=['main'])
parser.add_argument('--nepoch', type=int, default=50)
## environment
parser.add_argument("--duplicate", default='ask')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--tqdm', action='store_true')
parser.add_argument('--fp16', action='store_true')
## model
parser.add_argument('--structure', choices=structures, required=True)
parser.add_argument('--weight')

parser.add_argument('--bsz', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--weight-decay', type=float, default=0)
args, _ = parser.parse_known_args()
if args.optimizer == 'lars':
    # 特にdefaultはないが, vicreg, vicreglのパラメータから判断して
    parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--scheduler', default='cosine_annealing')

## Barlow Twins
parser.add_argument('--head-size', type=int, default=128)
parser.add_argument('--lambda-param', type=float, default=5e-3)
parser.add_argument('--resize-scale-min', type=float, default=0.08)
parser.add_argument('--resize-scale-max', type=float, default=1.0)
parser.add_argument('--resize-ratio-max', type=float, default=4/3)

args, _ = parser.parse_known_args()
if args.scheduler == 'cosine_annealing_warmup':
    parser.add_argument('--warmup', type=int, default=10) # default in VICRegL

args = parser.parse_args()
if hasattr(args, 'base_lr'):
    args.lr = args.base_lr * args.bsz / 256

weight = args.weight if args.weight in structure2weights[args.structure] else None
backbone = get_backbone(args.structure, weight)
model = BarlowTwins(backbone, args.lambda_param, args.head_size, 
                args.resize_scale_min, args.resize_scale_max, args.resize_ratio_max)

pretrain(args, model)
