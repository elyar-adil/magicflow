import abc

from mf.base.optimizer.optimizer import Optimizer


class OptimizerImpl(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.__optim = None

    def step(self):
        if not hasattr(self, '__optim') or self.__optim is None:
            self.__optim = self.make_optimizer(*self.args, **self.kwargs)
        self.__optim.step()

    def zero_grad(self):
        if self.__optim is None:
            self.__optim = self.make_optimizer(*self.args, **self.kwargs)
        self.__optim.zero_grad()

    @abc.abstractmethod
    def make_optimizer(self, *args, **kwargs):
        pass