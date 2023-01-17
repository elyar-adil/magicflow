import abc

from .module import Module


class Upsample(abc.ABC, Module):
    def __init__(self, size=None, scale_factor=None,
                 mode: str = 'nearest', align_corners=None):
        ...
