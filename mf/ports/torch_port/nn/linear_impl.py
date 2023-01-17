import torch

from mf.base.nn.linear import Linear
from mf.ports.torch_port.nn.module_impl import ModuleImpl


class LinearImpl(Linear, ModuleImpl):
    def __init__(self, out_channels, bias=True):
        Linear.__init__(self, out_channels, bias)
        ModuleImpl.__init__(self)

    def _make_block(self, in_channels, out_channels, bias):
        self.block = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.block = self.block.to(device=self.device)
