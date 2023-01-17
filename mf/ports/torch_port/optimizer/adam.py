import torch

from mf.base.optimizer.adam import Adam
from mf.ports.torch_port.optimizer.optimizer import OptimizerImpl


class AdamImpl(OptimizerImpl, Adam):
    def __init__(self, parameters_getter, learning_rate, weight_decay=0.0):
        Adam.__init__(self, parameters_getter, learning_rate, weight_decay=weight_decay)
        OptimizerImpl.__init__(self, parameters_getter, learning_rate, weight_decay=weight_decay)

    def make_optimizer(self, parameters_getter, learning_rate, weight_decay=0.0):
        return torch.optim.Adam(parameters_getter(), lr=learning_rate,
                                weight_decay=weight_decay)
