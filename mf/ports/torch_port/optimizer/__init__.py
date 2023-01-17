from mf.ports.torch_port.optimizer.adam import AdamImpl
from mf.ports.torch_port.optimizer.sgd import SGDImpl


SGD = SGDImpl
Adam = AdamImpl

__all__ = ["SGD", "Adam"]
