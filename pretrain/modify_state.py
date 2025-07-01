import sys, os
from glob import glob
from argparse import ArgumentParser
import torch
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/mtpc"]
from src.utils.logger import get_logger
from src.pretrain import get_scheme, get_argv
from src.model.state_dict import state_modifiers
from src.utils.time import wtqdm
logger = get_logger(stream=True)

for mpath in (pbar:=wtqdm(glob(f"{WORKDIR}/mtpc/pretrain/results/**/model.pth", recursive=True))):
    rdir = os.path.dirname(mpath)
    scheme = get_scheme(rdir)
    out_path = f"{rdir}/resnet50.pth"
    
    if scheme is None: continue
    if os.path.exists(out_path):
        continue

    # parse args & check epoch
    argv = get_argv(rdir)
    if scheme == 'VICRegL': argv = argv[1:]
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int)
    args, _ = parser.parse_known_args(argv)

    ## load state
    pbar.start('load_state')
    match scheme:
        case 'barlowtwins':
            state = torch.load(mpath, weights_only=True)
        case 'vicreg':
            def exclude_bias_and_norm(p):
                return p.ndim == 1
            state = torch.load(mpath)
        case 'VICRegL':
            sys.path.append(f"{WORKDIR}/mtpc/pretrain/VICRegL")
            state = torch.load(mpath)
            sys.path = sys.path[:-1]
        case _:
            raise ValueError(f"Unknown {scheme=}")

    ## check epoch
    pbar.start('check_epoch')
    epoch = state['epoch']
    if epoch != args.epochs:
        print(f"{mpath}({epoch}) is not checkpoint of final epoch({args.epochs}).")
        continue
    
    # modify
    pbar.start('save')
    state = state_modifiers[scheme](state)
    torch.save(state, out_path)
