import torch.nn

from mf.base.nn.norm import BatchNorm2D
from mf.ports.torch_port.nn.module_impl import ModuleImpl


class BatchNorm2DImpl(BatchNorm2D, ModuleImpl):
    def __init__(self):
        BatchNorm2D.__init__(self)
        ModuleImpl.__init__(self)

    def _make_block(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.block = torch.nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine)
        self.block = self.block.to(device=self.device)
