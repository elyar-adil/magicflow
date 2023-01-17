import abc

from mf.base.optimizer.optimizer import Optimizer


class Adam(Optimizer, abc.ABC):
    def __init__(self, parameters_getter, learning_rate, weight_decay=0.0):
        super().__init__()
