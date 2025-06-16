import sys, os
import torch
from torch import Tensor
from collections import OrderedDict
from collections.abc import Mapping, Callable

def bt_modifier(state: Mapping[str, torch.Tensor]):
    new_state = OrderedDict()
    for key, value in state.items():
        if key.startswith('head.'): continue
        assert key.startswith('backbone.')
        new_state[key[len('backbone.'):]] = value
    return new_state

def barlowtwins_modifier(state: Mapping[str, Tensor]):
    state = state['model']
    new_state = {}
    for key, value in state.items():
        if key.startswith('module.projector.') or key.startswith('module.bn.'): continue
        assert key.startswith('module.backbone.'), key
        key = key[len('module.backbone.'):]
        new_state[key] = value
    return new_state

def vicreg_modifier(state: dict):
    state = state['model']
    new_state = {}
    for key, value in state.items():
        if key.startswith('module.projector.'): continue
        assert key.startswith('module.backbone.'), key
        key = key[len('module.backbone.'):]
        new_state[key] = value
    return new_state

def vicregl_modifier(state: dict):
    state = state['model']
    new_state = {}
    for key, value in state.items():
        if key.startswith(('module.projector.', "module.classifier.", 'module.maps_projector.')): continue
        assert key.startswith('module.backbone.'), key
        key = key[len('module.backbone.'):]
        new_state[key] = value
    return new_state


state_modifiers: dict[str, Callable[[dict], dict]] = {
    'null': lambda x: x,
    'bt': bt_modifier, 
    'barlowtwins': barlowtwins_modifier,
    'vicreg': vicreg_modifier,
    'VICRegL': vicregl_modifier
}
