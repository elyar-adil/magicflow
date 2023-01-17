import paddle

from mf.base.nn.pooling import MaxPool2D, AvgPool2D
from mf.ports.paddle_port.nn.module_impl import ModuleImpl


class MaxPool2DImpl(MaxPool2D, ModuleImpl, paddle.nn.MaxPool2D):
    def __init__(self, kernel_size, stride=None, padding=0,
                 return_indices: bool = False, ceil_mode: bool = False):
        ModuleImpl.__init__(self)
        MaxPool2D.__init__(self, kernel_size, stride=stride, padding=padding,
                           return_indices=return_indices, ceil_mode=ceil_mode)
        paddle.nn.MaxPool2D.__init__(self, kernel_size, stride=stride, padding=padding,
                                     return_mask=return_indices, ceil_mode=ceil_mode)

    def forward(self, x):
        return x.__class__(__raw_tensor__=paddle.nn.MaxPool2D.forward(self, x.__raw_tensor__))


class AvgPool2DImpl(AvgPool2D, ModuleImpl, paddle.nn.AvgPool2D):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode: bool = False, count_include_pad: bool = True,
                 divisor_override=None):
        ModuleImpl.__init__(self)
        AvgPool2D.__init__(self, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                           count_include_pad=count_include_pad,
                           divisor_override=divisor_override)
        paddle.nn.AvgPool2D.__init__(self, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                                     exclusive=count_include_pad, divisor_override=divisor_override)

    def forward(self, x):
        return x.__class__(__raw_tensor__=paddle.nn.AvgPool2D.forward(self, x.__raw_tensor__))
