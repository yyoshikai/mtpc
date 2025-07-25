from glob import glob
import torch
from tqdm import tqdm

for path in tqdm(sorted(glob("results/**/resnet50.pth", recursive=True))):
    state = torch.load(path, weights_only=True, map_location='cpu')

    n_nan = 0
    n_param = 0
    param: torch.Tensor
    for param in state.values():
        n_nan += (~torch.isfinite(param)).sum().item()
        n_param += param.numel()
    
    if n_nan > 0:
        print(f"{path}: {n_nan}/{n_param}")

print(f"checked {len(paths)} weights.")
