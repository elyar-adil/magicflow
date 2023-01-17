import abc


class Optimizer(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def zero_grad(self):
        pass
