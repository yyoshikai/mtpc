import torch
import torch.nn as nn
import torchvision
from typing import Optional
from collections.abc import Callable
from torch import Tensor

from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torchvision.models.resnet import _resnet, conv3x3
from torch.nn.modules.module import _IncompatibleKeys

class BarlowTwins0(nn.Module):
    def __init__(self, from_resnet=False, head_size: int=128):
        super().__init__()
        backbone = resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            if from_resnet else None
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, head_size)
        )

    def forward(self, x):
        x = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.head(x)


class BasicBlock2(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

class BarlowTwins(nn.Module):
    def __init__(self, from_resnet=False, head_size: int=128):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if from_resnet else None
        backbone = _resnet(BasicBlock2, [2, 2, 2, 2], weights, progress=True)
        
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, head_size)
        )

    def forward(self, x):
        x = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.head(x)

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

class ResNetModel(nn.Module):
    def __init__(self, structure='resnet50', from_scratch=False, output_mean: float=0.0, output_std: float=1.0):
        super().__init__()
        match structure:
            case 'resnet50':
                backbone = resnet50(weights=None if from_scratch else ResNet50_Weights.IMAGENET1K_V2)
                out_size = 2048
            case 'resnet18':
                backbone = _resnet(BasicBlock2, [2, 2, 2, 2], weights=None if from_scratch else ResNet18_Weights.IMAGENET1K_V1, progress=True)
                out_size = 512
            case _: 
                raise ValueError
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(out_size, 128),
            nn.GELU(),
            nn.Linear(128, 1))
        
        self.register_buffer('output_mean', torch.tensor(output_mean))
        self.register_buffer('output_std', torch.tensor(output_std))
        self.register_load_state_dict_post_hook(fill_output_mean_std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x.squeeze_(-1, -2)
        x = self.head(x)
        x.squeeze_(-1)
        return x*self.output_std+self.output_mean
