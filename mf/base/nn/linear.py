import abc

from .module import Module


class Linear(abc.ABC, Module):
    def __init__(self, out_channels, bias=True):
        super().__init__()
        self.block = None
        self.in_channels = None
        self.out_channels = out_channels
        self.bias = bias

    def forward(self, x):
        tensor = x.__raw_tensor__
        if self.block is None:
            self.in_channels = tensor.shape[-1]
            self._make_block(self.in_channels, self.out_channels, bias=self.bias)
        return x.__class__(__raw_tensor__=self.block(tensor))

    @abc.abstractmethod
    def _make_block(self, in_channels, out_channels, bias):
        pass
