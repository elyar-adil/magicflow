import torch

from mf.base.optimizer.sgd import SGD
from mf.ports.torch_port.optimizer.optimizer import OptimizerImpl


class SGDImpl(OptimizerImpl, SGD):
    def __init__(self, parameters, learning_rate, weight_decay=0.0):
        SGD.__init__(self, parameters, learning_rate, weight_decay=weight_decay)
        OptimizerImpl.__init__(self, parameters, learning_rate, weight_decay=weight_decay)

    def make_optimizer(self, parameters_getter, learning_rate, weight_decay=0.0):
        return torch.optim.Adam(parameters_getter(), lr=learning_rate,
                                weight_decay=weight_decay)
