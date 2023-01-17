import paddle

from mf.base.nn.linear import Linear
from mf.ports.paddle_port.nn.module_impl import ModuleImpl


class LinearImpl(Linear, ModuleImpl):
    def __init__(self, out_channels, bias=True):
        ModuleImpl.__init__(self)
        Linear.__init__(self, out_channels, bias)

    def _make_block(self, in_channels, out_channels, bias: bool):
        self.block = paddle.nn.Linear(in_channels, out_channels, bias_attr=bias, name="Linear")
        self.block = self.block.to(device=self.device)
