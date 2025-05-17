from typing import Optional, Callable, Any
from argparse import ArgumentParser, Namespace
from PIL.Image import Image
import torch.nn as nn
from torch import Tensor
from .backbone import Backbone

class Scheme(nn.Module):
    backbone: Backbone

    def get_train_transform(self, example_dir: Optional[str]=None, n_example: int=0) -> Callable[[Image], Any]:
        raise NotImplementedError
    def get_eval_transform(self) -> Callable[[Image], Tensor]:
        raise NotImplementedError
    @classmethod
    def add_args(parser: ArgumentParser) -> None:
        raise NotImplementedError
    @classmethod
    def from_args(cls, args: Namespace, backbone: Backbone) -> 'Scheme':
        raise NotImplementedError
