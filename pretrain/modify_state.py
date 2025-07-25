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

for mpath in sorted(glob(f"./results/**/model.pth", recursive=True)):
    rdir = os.path.dirname(mpath)
    scheme = get_scheme(rdir)
    out_path = f"{rdir}/resnet50.pth"
    
    if scheme != 'VICRegL': continue
    if os.path.exists(out_path):
        continue

    # parse args & check epoch
    argv = get_argv(rdir)[1:]
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int)
    args, _ = parser.parse_known_args(argv)

    ## load state
    sys.path.append(f"{WORKDIR}/mtpc/pretrain/VICRegL")
    state = torch.load(mpath)
    sys.path = sys.path[:-1]

    ## check epoch
    epoch = state['epoch']
    if epoch != args.epochs:
        logger.info(f"Not final checkpoint: {mpath}({epoch})")
        continue
    
    # modify
    logger.info(f"Modified: {mpath}({epoch})")
    state = state_modifiers[scheme](state)
    torch.save(state, out_path)
