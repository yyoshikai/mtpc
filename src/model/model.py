import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
from .backbone import Backbone

class BarlowTwins(nn.Module):
    def __init__(self, backbone: Backbone, lambda_param: float, 
            gather_distributed: bool=False, head_size: int=128):
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
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

    def forward(self, x_a: Tensor, x_b: Tensor):

        # model
        z_a: Tensor = self.head(self.backbone(x_a))
        z_b: Tensor = self.head(self.backbone(x_b))

        # criterion
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

class VICReg(nn.Module):
    """
    from https://github.com/facebookresearch/vicreg
    """
    def __init__(self, backbone: Backbone, head_sizes: list[int], 
                sim_coeff, std_coeff, cov_coeff):
        super().__init__()
        self.backbone = backbone
        head_sizes = [backbone.output_size]+head_sizes
        head_layers = []
        for i in range(len(head_sizes)-2):
            head_layers += [
                nn.Linear(head_sizes[i], head_sizes[i+1]),
                nn.BatchNorm1d(head_sizes[i+1]),
                nn.ReLU(True)
            ]
        head_layers.append(nn.Linear(head_sizes[-2], head_sizes[-1], bias=False))
        self.head = nn.Sequential(*head_layers)
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x_a: Tensor, x_b: Tensor):
        x: Tensor = self.head(self.backbone(x_a))
        y: Tensor = self.head(self.backbone(x_b))
        B, D = x.shape

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (B - 1)
        cov_y = (y.T @ y) / (B - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(D) \
                + off_diagonal(cov_y).pow_(2).sum().div(D)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

