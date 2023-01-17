import abc

import torch

from mf.base.nn.module import Module


class ModuleImpl(Module, torch.nn.Module):
    def __init__(self):
        Module.__init__(self)
        torch.nn.Module.__init__(self)
        self.device = None

    @abc.abstractmethod
    def forward(self, *inputs, **kwargs):
        pass

    def parameters(self, recurse: bool = True):
        return torch.nn.Module.parameters(self, recurse=recurse)

    def __call__(self, *args, **kwargs):
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def to(self, device=None):
        self.device = device
        if device is not None:
            for child in self.children():
                child.to(device)
            return torch.nn.Module.to(self, device)
        return self
