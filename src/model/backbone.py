from typing import Optional
from collections.abc import Callable
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet, vision_transformer as vit, convnext, WeightsEnum


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
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = resnet.conv3x3(planes, planes)
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

class Backbone(nn.Module):
    structure2params: dict[str, dict]
    structure2weights: dict[str, dict[str, WeightsEnum]]

    @classmethod
    def structures(cls):
        return cls.structure2params.keys()

    @property
    def output_size(self) -> int:
        raise NotImplementedError

    def get_pos_feature(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def predict(self, x: Tensor) -> Tensor:
        raise NotImplementedError

class ResNet(resnet.ResNet, Backbone):
    structure2params = {
        'resnet18': dict(block=BasicBlock2, layers=[2, 2, 2, 2]), 
        'resnet34': dict(block=BasicBlock2, layers=[3, 4, 6, 3]), 
        'resnet50': dict(block=resnet.Bottleneck, layers=[3, 4, 6, 3]),
        'resnet101': dict(block=resnet.Bottleneck, layers=[3, 4, 23, 3]),
        'resnet152': dict(block=resnet.Bottleneck, layers=[3, 8, 36, 3]),
    }

    structure2weights = {
        'resnet18': {'imagenet': resnet.ResNet18_Weights.IMAGENET1K_V1}, 
        'resnet34': {'imagenet': resnet.ResNet34_Weights.IMAGENET1K_V1}, 
        'resnet50': {'imagenet': resnet.ResNet50_Weights.IMAGENET1K_V2}, 
        'resnet101': {'imagenet': resnet.ResNet101_Weights.IMAGENET1K_V2}, 
        'resnet152': {'imagenet': resnet.ResNet152_Weights.IMAGENET1K_V2}, 
    }

    def __init__(self, structure: str, weight: str|None):
        num_classes = 1000
        self.structure = structure
        if weight is not None:
            weight = self.structure2weights[structure][weight]
            num_classes = len(weight.meta['categories'])

        super().__init__(**self.structure2params[structure], num_classes=num_classes)
        if weight is not None:
            self.load_state_dict(weight.get_state_dict(progress=True, check_hash=True))

    def get_pos_feature(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: Tensor):
        x = self.get_pos_feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def predict(self, x: Tensor):
        return self.fc(self(x))

    @property    
    def output_size(self):
        match self.structure:
            case 'resnet50': return 2048
            case 'resnet18': return 512

class ViT(vit.VisionTransformer, Backbone):
    structure2params = {
        'vit_b_16': dict(patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072,),
        'vit_b_32': dict(patch_size=32, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072,),
        'vit_l_16': dict(patch_size=16, num_layers=24, num_heads=16, hidden_dim=1024, mlp_dim=4096,),
        'vit_l_32': dict(patch_size=32, num_layers=24, num_heads=16, hidden_dim=1024, mlp_dim=4096,),
    }
    structure2weights = {
        'vit_b_16': {'imagenet': vit.ViT_B_16_Weights.IMAGENET1K_V1},
        'vit_b_32': {'imagenet': vit.ViT_B_32_Weights.IMAGENET1K_V1},
        'vit_l_16': {'imagenet': vit.ViT_L_16_Weights.IMAGENET1K_V1},
        'vit_l_32': {'imagenet': vit.ViT_L_32_Weights.IMAGENET1K_V1},
    }

    def __init__(self, structure: str, weight: str|None):
        num_classes = 1000
        image_size = 224
        if weight is not None:
            weight: WeightsEnum = self.structure2weights[structure][weight]
            num_classes = len(weight.meta['categories'])
            image_size = weight.meta['min_size'][0]

        super().__init__(**self.structure2params[structure], 
                num_classes=num_classes, image_size=image_size)
        if weight is not None:
            self.load_state_dict(weight.get_state_dict(progress=True, check_hash=True))

    @property
    def output_size(self):
        return self.hidden_dim
    
    def _base_forward(self, x: Tensor) -> Tensor:
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        return x
    
    def forward(self, x: Tensor):
        x = self._base_forward(x) # [B, H*W+1, C]
        return x[:, 0] # get class token

    def get_pos_feature(self, x: Tensor) -> Tensor:
        # Reshape and permute the input tensor
        n, c, h, w = x.shape
        n_h = h // self.patch_size
        n_w = w // self.patch_size

        x = self._base_forward(x)
        x = x[:, 1:].reshape(n, n_h, n_w, self.output_size).permute(0, 3, 1, 2)

        return x

    def predict(self, x: Tensor):
        return super().forward(x)

class ConvNeXt(convnext.ConvNeXt, Backbone):
    structure2params = {
        'convnext_tiny': dict(
            block_setting=[
                convnext.CNBlockConfig(96, 192, 3),
                convnext.CNBlockConfig(192, 384, 3),
                convnext.CNBlockConfig(384, 768, 9),
                convnext.CNBlockConfig(768, None, 3),
            ], 
            stochastic_depth_prob=0.1
        ), 
        'convnext_small': dict(
            block_setting = [
                convnext.CNBlockConfig(96, 192, 3),
                convnext.CNBlockConfig(192, 384, 3),
                convnext.CNBlockConfig(384, 768, 27),
                convnext.CNBlockConfig(768, None, 3),
            ],
            stochastic_depth_prob=0.4
        ), 
        'convnext_base': dict(block_setting = [
                convnext.CNBlockConfig(128, 256, 3),
                convnext.CNBlockConfig(256, 512, 3),
                convnext.CNBlockConfig(512, 1024, 27),
                convnext.CNBlockConfig(1024, None, 3),
            ],
            stochastic_depth_prob=0.5
        ), 
        'convnext_large': dict(block_setting = [
                convnext.CNBlockConfig(192, 384, 3),
                convnext.CNBlockConfig(384, 768, 3),
                convnext.CNBlockConfig(768, 1536, 27),
                convnext.CNBlockConfig(1536, None, 3),
            ],
            stochastic_depth_prob=0.5
        ), 
    }

    structure2weights = {
        'convnext_tiny': {'imagenet': convnext.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,},
        'convnext_small': {'imagenet': convnext.ConvNeXt_Small_Weights.IMAGENET1K_V1,},
        'convnext_base': {'imagenet': convnext.ConvNeXt_Base_Weights.IMAGENET1K_V1,},
        'convnext_large': {'imagenet': convnext.ConvNeXt_Base_Weights.IMAGENET1K_V1,},
    }

    def __init__(self, structure: str, weight):
        num_classes = 1000
        if weight is not None:
            weight = self.structure2weights[structure][weight]
            num_classes = len(weight.meta['categories'])
        super().__init__(**self.structure2params[structure], num_classes=num_classes)
        if weight is not None:
            self.load_state_dict(weight.get_state_dict(progress=True, check_hash=True))
        self._output_size = self.classifier[0].normalized_shape[0]



    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        return self.avgpool(self.features(x)).reshape(B, -1)

    def get_pos_feature(self, x: Tensor):
        return self.features(x)

    def predict(self, x: Tensor):
        return super().forward(x)

    @property
    def output_size(self):
        return self.classifier[0].normalized_shape[0]

backbone_clss: list[type] = [ResNet, ViT, ConvNeXt]
structures = []
structure2weights = {}
for cls in backbone_clss:
    structures += cls.structures()
    for structure, weights in cls.structure2weights.items():
        structure2weights[structure] = list(weights.keys())

def get_backbone(structure: str, weight = None) -> Backbone:
    for cls in backbone_clss:
        if structure in cls.structures():
            return cls(structure, weight)
    raise ValueError(f"Unsupported {structure=}")
