import abc

from .module import Module


class ReLU(abc.ABC, Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass


class Softmax(abc.ABC, Module):
    def __init__(self, dim=None):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass
