import abc

import paddle

from mf.base.nn.module import Module


class ModuleImpl(Module, paddle.nn.Layer):
    def __init__(self):
        Module.__init__(self)
        paddle.nn.Layer.__init__(self)
        self.device = None

    @abc.abstractmethod
    def forward(self, *inputs, **kwargs):
        pass

    def parameters(self, recurse: bool = True):
        return paddle.nn.Layer.parameters(self, include_sublayers=recurse)

    def __call__(self, *args, **kwargs):
        return paddle.nn.Layer.__call__(self, *args, **kwargs)

    def to(self, device=None, dtype=None, blocking=None):
        if device == 'cuda':
            device = 'gpu'
        self.device = device

        if device:
            for child in self.children():
                child.to(device)
            return paddle.nn.Layer.to(self, device=device)
        else:
            return self
