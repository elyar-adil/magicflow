import torch

from mf.base.nn.upsampling import Upsample
from mf.ports.torch_port.nn.module_impl import ModuleImpl


class UpsampleImpl(Upsample, ModuleImpl, torch.nn.Upsample):
    def __init__(self, size=None, scale_factor=None,
                 mode: str = 'nearest', align_corners=None):
        ModuleImpl.__init__(self)
        Upsample.__init__(self, size=size, scale_factor=scale_factor,
                          mode=mode, align_corners=align_corners)
        torch.nn.Upsample.__init__(self, size=size, scale_factor=scale_factor,
                                   mode=mode, align_corners=align_corners)

    def forward(self, x):
        return x.__class__(__raw_tensor__=torch.nn.Upsample.forward(self, x.__raw_tensor__))
