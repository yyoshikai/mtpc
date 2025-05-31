from collections.abc import Callable
from argparse import ArgumentParser, Namespace
import numpy as np
from PIL.Image import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..backbone import Backbone
from .transforms import VICRegLTransform

class VICRegL(nn.Module):
    """
    TODO: add license
    """
    def __init__(self, backbone: Backbone, head_sizes: list[int], 
                map_head_sizes: list[int], 
                alpha: float, head_norm_layer: str, 
                inv_coeff, var_coeff, cov_coeff, 
                l2_all_matches: bool, # True
                num_matches: list[int], # [20, 4] 
                fast_vc_reg: bool, # False
                
                size_crops: list[int], num_crops: list[int],
                min_scale_crops: list[float], max_scale_crops: list[float],
                no_flip_grid: bool
        ):
        super().__init__()
        self.embedding_dim = int(head_sizes[-1])

        self.backbone = backbone
        self.backbone_size = backbone.output_size
        self.alpha = alpha

        if self.alpha < 1.0:
            self.maps_projector = mlp([self.backbone_size]+map_head_sizes, 
                    head_norm_layer)

        if self.alpha > 0.0:
            self.projector = mlp([self.backbone_size]+head_sizes, head_norm_layer)

        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.l2_all_matches = l2_all_matches
        self.num_matches = num_matches
        self.fast_vc_reg = fast_vc_reg

        self.size_crops = size_crops
        self.num_crops = num_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops
        self.no_flip_grid = no_flip_grid
        # self.classifier = nn.Linear(self.backbone_size, self.args.num_classes)

    def _vicreg_loss(self, x: Tensor, y: Tensor):
        repr_loss = self.inv_coeff * F.mse_loss(x, y)

        x = x - x.mean(dim=0) # gather_center
        y = y - y.mean(dim=0) # gather_center

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = self.var_coeff * (
            torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
        )

        x = x.permute((1, 0, 2))
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = NC
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
            cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean()
        cov_loss = self.cov_coeff * cov_loss

        return repr_loss, std_loss, cov_loss

    def _local_loss(self, maps_1, maps_2, location_1, location_2):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        # L2 distance based bacthing
        if self.l2_all_matches:
            num_matches_on_l2 = [None, None]
        else:
            num_matches_on_l2 = self.num_matches

        maps_1_filtered, maps_1_nn = neirest_neighbores_on_l2(
            maps_1, maps_2, num_matches=num_matches_on_l2[0]
        )
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_l2(
            maps_2, maps_1, num_matches=num_matches_on_l2[1]
        )

        if self.fast_vc_reg:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        # Location based matching
        location_1 = location_1.flatten(1, 2)
        location_2 = location_2.flatten(1, 2)

        maps_1_filtered, maps_1_nn = neirest_neighbores_on_location(
            location_1,
            location_2,
            maps_1,
            maps_2,
            num_matches=self.num_matches[0],
        )
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_location(
            location_2,
            location_1,
            maps_2,
            maps_1,
            num_matches=self.num_matches[1],
        )

        if self.fast_vc_reg:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        return inv_loss, var_loss, cov_loss

    def local_loss(self, maps_embedding, locations):
        num_views = len(maps_embedding)
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss_this, var_loss_this, cov_loss_this = self._local_loss(
                    maps_embedding[i], maps_embedding[j], locations[i], locations[j],
                )
                inv_loss = inv_loss + inv_loss_this
                var_loss = var_loss + var_loss_this
                cov_loss = cov_loss + cov_loss_this
                iter_ += 1

        if self.fast_vc_reg:
            inv_loss = self.inv_coeff * inv_loss / iter_
            var_loss = 0.0
            cov_loss = 0.0
            iter_ = 0
            for i in range(num_views):
                x = maps_embedding[i] - maps_embedding[i].mean(dim=0) # gather_center
                std_x = torch.sqrt(x.var(dim=0) + 0.0001)
                var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
                x = x.permute(1, 0, 2)
                *_, sample_size, num_channels = x.shape
                non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
                x = x - x.mean(dim=-2, keepdim=True)
                cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
                cov_loss = cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels
                cov_loss = cov_loss + cov_loss.mean()
                iter_ = iter_ + 1
            var_loss = self.var_coeff * var_loss / iter_
            cov_loss = self.cov_coeff * cov_loss / iter_
        else:
            inv_loss = inv_loss / iter_
            var_loss = var_loss / iter_
            cov_loss = cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def global_loss(self, embedding, maps=False):
        num_views = len(embedding)
        inv_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
                iter_ = iter_ + 1
        inv_loss = self.inv_coeff * inv_loss / iter_

        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(num_views):
            x = embedding[i] - embedding[i].mean(dim=0)
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + off_diagonal(cov_x).pow_(2).sum().div(
                self.embedding_dim
            )
            iter_ = iter_ + 1
        var_loss = self.var_coeff * var_loss / iter_
        cov_loss = self.cov_coeff * cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def compute_metrics(self, outputs):
        def correlation_metric(x):
            x_centered = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-05)
            return torch.mean(
                off_diagonal((x_centered.T @ x_centered) / (x.size(0) - 1))
            )

        def std_metric(x):
            x = F.normalize(x, p=2, dim=1)
            return torch.mean(x.std(dim=0))

        representation = outputs["representation"][0].contiguous() # batch_all_gather
        corr = correlation_metric(representation)
        stdrepr = std_metric(representation)

        if self.alpha > 0.0:
            embedding = outputs["embedding"][0].contiguous() # batch_all_gather
            core = correlation_metric(embedding)
            stdemb = std_metric(embedding)
            return dict(stdr=stdrepr, stde=stdemb, corr=corr, core=core)

        return dict(stdr=stdrepr, corr=corr)

    def forward_networks(self, inputs, is_val):        
        outputs = {
            "representation": [],
            "embedding": [],
            "maps_embedding": [],
            "logits": [],
            "logits_val": [],
        }


        for x in inputs["views"]:
            maps, representation = self.backbone.get_pos_feature(x)
            outputs["representation"].append(representation)

            if self.alpha > 0.0:
                embedding = self.projector(representation)
                outputs["embedding"].append(embedding)

            if self.alpha < 1.0:
                # batch_size, num_loc, _ = maps.shape
                B, C, H, W = maps.shape
                maps = maps.view(B, C, H*W).permute(0, 2, 1)

                maps_embedding = self.maps_projector(maps.flatten(start_dim=0, end_dim=1))
                maps_embedding = maps_embedding.view(B, H*W, -1)
                outputs["maps_embedding"].append(maps_embedding)

            # logits = self.classifier(representation.detach())
            # outputs["logits"].append(logits)

        if is_val:
            _, representation = self.backbone(inputs["val_view"])
            val_logits = self.classifier(representation.detach())
            outputs["logits_val"].append(val_logits)

        return outputs

    def forward(self, inputs, is_val=False, backbone_only=False):
        # inputs: {'views': list[Tensor(B, H, W, C?)], 'locations': list[Tensor(B,  G, G, 2)]], 'labels': Tensor(B)} G=grid_size
        device = next(self.parameters()).device
        inputs = {
            'views': [x.to(device) for x in inputs['views']], 
            'locations': [x.to(device) for x in inputs['locations']]
        }

        if backbone_only:
            maps, _ = self.backbone(inputs)
            return maps

        outputs = self.forward_networks(inputs, is_val)
        with torch.no_grad():
            logs = self.compute_metrics(outputs)
        loss = 0.0

        # Global criterion
        if self.alpha > 0.0:
            inv_loss, var_loss, cov_loss = self.global_loss(
                outputs["embedding"]
            )
            loss = loss + self.alpha * (inv_loss + var_loss + cov_loss)
            logs.update(dict(inv_l=inv_loss, var_l=var_loss, cov_l=cov_loss,))

        # Local criterion
        # Maps shape: B, C, H, W
        # With convnext actual maps shape is: B, H * W, C
        if self.alpha < 1.0:
            (
                maps_inv_loss,
                maps_var_loss,
                maps_cov_loss,
            ) = self.local_loss(
                outputs["maps_embedding"], inputs["locations"]
            )
            loss = loss + (1 - self.alpha) * (
                maps_inv_loss + maps_var_loss + maps_cov_loss
            )
            logs.update(
                dict(minv_l=maps_inv_loss, mvar_l=maps_var_loss, mcov_l=maps_cov_loss,)
            )

        # Online classification
        """
        labels = inputs["labels"]
        classif_loss = F.cross_entropy(outputs["logits"][0], labels)
        acc1, acc5 = utils.accuracy(outputs["logits"][0], labels, topk=(1, 5))
        loss = loss + classif_loss
        logs.update(dict(cls_l=classif_loss, top1=acc1, top5=acc5, l=loss))
        if is_val:
            classif_loss_val = F.cross_entropy(outputs["logits_val"][0], labels)
            acc1_val, acc5_val = utils.accuracy(
                outputs["logits_val"][0], labels, topk=(1, 5)
            )
            logs.update(
                dict(clsl_val=classif_loss_val, top1_val=acc1_val, top5_val=acc5_val,)
            )
        """

        return loss # , logs
    
    def get_train_transform(self, example_dir, n_example) -> Callable[[Image], dict[str, Tensor]]:
        return VICRegLTransform(example_dir, n_example, self.size_crops, self.num_crops, self.min_scale_crops, self.max_scale_crops, self.no_flip_grid)

    def get_eval_transform(self):
        raise NotImplementedError

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0)

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False)
    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)

    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )
    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

    indices = (
        torch.arange(0, topk_values.size(1))
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(topk_values.device)
    )
    indices_selected = indices.masked_select(mask)
    indices_selected = indices_selected.reshape(batch_size, num_matches)

    filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    return filtered_input_maps, filtered_candidate_maps

def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_maps, candidate_maps)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)

def neirest_neighbores_on_location(
    input_location, candidate_location, input_maps, candidate_maps, num_matches
):
    """
    input_location: (B, H * W, 2)
    candidate_location: (B, H * W, 2)
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_location, candidate_location)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)

def mlp(sizes: int, norm: str):
    layers = []
    for i in range(len(sizes)-2):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if norm == 'batch_norm': 
            layers.append(nn.BatchNorm1d(sizes[i+1]))
        elif norm == 'layer_norm':
            layers.append(nn.LayerNorm(sizes[i+1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    return nn.Sequential(*layers)
