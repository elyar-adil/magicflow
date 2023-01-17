import paddle

from mf.base.nn.container import Sequential
from mf.ports.paddle_port.nn.module_impl import ModuleImpl


class SequentialImpl(Sequential, ModuleImpl, paddle.nn.Sequential):
    def __init__(self, *args):
        ModuleImpl.__init__(self)
        Sequential.__init__(self, *args)
        paddle.nn.Sequential.__init__(self, *args)

    def forward(self, *inputs, **kwargs):
        return paddle.nn.Sequential.forward(self, *inputs, **kwargs)
