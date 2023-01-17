from mf.ports.paddle_port.optimizer.adam import AdamImpl
from mf.ports.paddle_port.optimizer.sgd import SGDImpl

SGD = SGDImpl
Adam = AdamImpl

__all__ = ["SGD", "Adam"]
