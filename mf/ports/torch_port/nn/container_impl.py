import torch

from mf.base.nn.container import Sequential
from mf.ports.torch_port.nn.module_impl import ModuleImpl


class SequentialImpl(Sequential, ModuleImpl, torch.nn.Sequential):
    def __init__(self, *args):
        ModuleImpl.__init__(self)
        Sequential.__init__(self, *args)
        torch.nn.Sequential.__init__(self, *args)

    def forward(self, *inputs, **kwargs):
        return torch.nn.Sequential.forward(self, *inputs, **kwargs)
