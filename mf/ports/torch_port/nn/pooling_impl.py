import torch

from mf.base.nn.pooling import MaxPool2D, AvgPool2D
from mf.ports.torch_port.nn.module_impl import ModuleImpl


class MaxPool2DImpl(MaxPool2D, ModuleImpl, torch.nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0,
                 return_indices: bool = False, ceil_mode: bool = False):
        MaxPool2D.__init__(self, kernel_size, stride=stride, padding=padding,
                           return_indices=return_indices, ceil_mode=ceil_mode)
        torch.nn.MaxPool2d.__init__(self, kernel_size, stride=stride, padding=padding,
                                    return_indices=return_indices, ceil_mode=ceil_mode)
        ModuleImpl.__init__(self)

    def forward(self, x):
        return x.__class__(__raw_tensor__=torch.nn.MaxPool2d.forward(self, x.__raw_tensor__))


class AvgPool2DImpl(AvgPool2D, ModuleImpl, torch.nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode: bool = False, count_include_pad: bool = True,
                 divisor_override=None):
        ModuleImpl.__init__(self)
        AvgPool2D.__init__(self, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                           count_include_pad=count_include_pad,
                           divisor_override=divisor_override)
        torch.nn.AvgPool2d.__init__(self, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                                    count_include_pad=count_include_pad, divisor_override=divisor_override)

    def forward(self, x):
        return x.__class__(__raw_tensor__=torch.nn.AvgPool2d.forward(self, x.__raw_tensor__))
