import abc

from .module import Module


class MaxPool2D(abc.ABC, Module):
    def __init__(self, kernel_size, stride=None, padding=0, return_indices: bool = False, ceil_mode: bool = False):
        ...


class AvgPool2D(abc.ABC, Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode: bool = False, count_include_pad: bool = True,
                 divisor_override=None):
        ...
