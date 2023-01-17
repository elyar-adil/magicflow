import paddle

from mf.base.nn.norm import BatchNorm2D
from mf.ports.paddle_port.nn.module_impl import ModuleImpl


class BatchNorm2DImpl(BatchNorm2D, ModuleImpl):
    def __init__(self):
        BatchNorm2D.__init__(self)
        ModuleImpl.__init__(self)

    def _make_block(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.block = paddle.nn.BatchNorm2D(num_features=num_features, epsilon=eps, momentum=momentum,
                                           weight_attr=None if affine else False, bias_attr=None if affine else False)
        self.block = self.block.to(device=self.device)
