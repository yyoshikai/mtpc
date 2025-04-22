import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
from .backbone import Backbone

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

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x)).squeeze(-1)*self.output_std+self.output_mean
