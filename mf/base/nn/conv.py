import abc

from .module import Module


class ConvBase(abc.ABC, Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros"):
        super().__init__()
        self.block = None

        self.in_channels = None
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

    def forward(self, x):
        tensor = x.__raw_tensor__
        if self.block is None:
            self.in_channels = tensor.shape[1]  # 'BCWH'
            self._make_block(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride,
                             padding=self.stride,
                             dilation=self.dilation, groups=self.groups, bias=self.bias,
                             padding_mode=self.padding_mode)
        return x.__class__(__raw_tensor__=self.block(tensor))

    @abc.abstractmethod
    def _make_block(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                    padding_mode):
        ...


class Conv2D(ConvBase):
    def _make_block(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                    padding_mode):
        raise NotImplementedError


class ConvTranspose2D(ConvBase):
    def _make_block(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                    padding_mode):
        raise NotImplementedError
