import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from .backbone import Backbone

class BarlowTwins(nn.Module):
    def __init__(self, backbone: Backbone, head_size: int=128):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(backbone.output_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, head_size)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

class BarlowTwinsCriterion(nn.Module):
    def __init__(
        self, lambda_param: float, gather_distributed : bool = False):
        super().__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:

        device = z_a.device

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD
        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        
        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss

def fill_output_mean_std(module: nn.Module, incompatible_keys: _IncompatibleKeys):
    if 'output_mean' in incompatible_keys.missing_keys:
        module.register_buffer('output_mean', torch.tensor(0.0))
        incompatible_keys.missing_keys.remove('output_mean')
    if 'output_std' in incompatible_keys.missing_keys:
        module.register_buffer('output_std', torch.tensor(1.0))
        incompatible_keys.missing_keys.remove('output_std')

class PredictModel(nn.Module):
    def __init__(self, backbone: Backbone, output_mean: float=0.0, output_std: float=1.0):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(backbone.output_size, 128),
            nn.GELU(),
            nn.Linear(128, 1))
        
        self.register_buffer('output_mean', torch.tensor(output_mean))
        self.register_buffer('output_std', torch.tensor(output_std))
        self.register_load_state_dict_post_hook(fill_output_mean_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)*self.output_std+self.output_mean
