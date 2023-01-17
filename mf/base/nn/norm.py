import abc

from .module import Module


class BatchNorm2D(abc.ABC, Module):

    def __init__(self, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.block = None
        self.num_features = None
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

    def forward(self, x):
        tensor = x.__raw_tensor__
        if self.block is None:
            self.num_features = tensor.shape[1] # 'BCWH'
            self._make_block(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine)
        return x.__class__(__raw_tensor__=self.block(tensor))

    @abc.abstractmethod
    def _make_block(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        pass
