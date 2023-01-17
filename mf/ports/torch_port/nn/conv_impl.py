import torch

from mf.base.nn.conv import Conv2D, ConvTranspose2D
from mf.ports.torch_port.nn.module_impl import ModuleImpl


class Conv2DImpl(Conv2D, ModuleImpl):

    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros"):
        ModuleImpl.__init__(self)
        Conv2D.__init__(self, out_channels=out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                        padding_mode=padding_mode)

    def _make_block(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                    padding_mode):
        self.block = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                     padding_mode=padding_mode)
        self.block = self.block.to(device=self.device)


class ConvTranspose2DImpl(ConvTranspose2D, ModuleImpl):

    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros"):
        ModuleImpl.__init__(self)
        ConvTranspose2D.__init__(self, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                 padding_mode=padding_mode)

    def _make_block(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                    padding_mode):
        self.block = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                                              bias=bias,
                                              padding_mode=padding_mode)
        self.block = self.block.to(device=self.device)
