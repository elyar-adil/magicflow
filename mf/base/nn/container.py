import abc

from .module import Module


class Sequential(abc.ABC, Module):
    def __init__(self, *args):
        super().__init__()
