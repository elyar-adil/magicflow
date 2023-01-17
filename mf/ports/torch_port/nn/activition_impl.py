import torch.nn.functional as f

from mf.base.nn.activition import ReLU, Softmax
from mf.ports.torch_port.nn.module_impl import ModuleImpl
from mf.ports.torch_port.tensor_impl import TensorImpl


class ReLUImpl(ReLU, ModuleImpl):
    def __init__(self):
        ModuleImpl.__init__(self)
        ReLU.__init__(self)

    def forward(self, x):
        return TensorImpl(__raw_tensor__=f.relu(x.__raw_tensor__))


class SoftmaxImpl(Softmax, ModuleImpl):
    def __init__(self, dim=None):
        ModuleImpl.__init__(self)
        Softmax.__init__(self, dim)
        self.dim = dim

    def forward(self, x):
        return TensorImpl(__raw_tensor__=f.softmax(x.__raw_tensor__, dim=self.dim))
