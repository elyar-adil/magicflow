import paddle

from mf.base.optimizer.sgd import SGD
from mf.ports.paddle_port.optimizer.optimizer import OptimizerImpl


class SGDImpl(OptimizerImpl, SGD):

    def __init__(self, parameters_getter, learning_rate, weight_decay=0.0):
        SGD.__init__(self, parameters_getter, learning_rate, weight_decay=weight_decay)
        OptimizerImpl.__init__(self, parameters_getter, learning_rate, weight_decay=weight_decay)

    def make_optimizer(self, parameters_getter, learning_rate, weight_decay=0.0):
        return paddle.optimizer.SGD(parameters=parameters_getter(), learning_rate=learning_rate,
                                    weight_decay=weight_decay)
