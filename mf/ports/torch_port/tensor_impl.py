import torch

from mf.base.tensor import Tensor


class TensorImpl(Tensor):

    def __init__(self, *args, **kwargs):
        if "__raw_tensor__" in kwargs.keys():
            super().__init__(kwargs["__raw_tensor__"])
            return
        super().__init__(torch.Tensor(*args, **kwargs))

    def __getattr__(self, item):
        if item == "shape":
            shape = self.__raw_tensor__.__getattribute__("shape")
            return [s for s in shape]
        return Tensor.__getattr__(self, item)
