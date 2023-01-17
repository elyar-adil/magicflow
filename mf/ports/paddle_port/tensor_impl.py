import inspect

import paddle

import mf.common.tensor_wrapping as tw
from mf.base.tensor import Tensor


class TensorImpl(Tensor):

    def __init__(self, *args, **kwargs):
        if "__raw_tensor__" in kwargs.keys():
            super().__init__(kwargs["__raw_tensor__"])
            return
        super().__init__(paddle.Tensor(*args, **kwargs))

    def __getattr__(self, item):
        if item == "requires_grad":
            return not self.__raw_tensor__.stop_gradient
        elif item == "to":
            def identity(*_args):
                return self

            return identity
        elif item == "long":
            def to_long():
                return tw.wrap_tensor(tw.unwrap_tensor(self)._to(dtype="int64"), TensorImpl)

            return to_long
        elif item == "float":
            def to_float():
                return tw.wrap_tensor(tw.unwrap_tensor(self)._to(dtype="float32"), TensorImpl)

            return to_float

        attr = self.__raw_tensor__.__getattribute__(item)
        if inspect.isbuiltin(attr) or inspect.ismethod(attr):
            def func(*args, **kwargs):
                if 'dim' in kwargs.keys():
                    kwargs['axis'] = kwargs['dim']
                    del kwargs['dim']
                result = attr(*args, **kwargs)
                if isinstance(result, self.__raw_tensor__.__class__):
                    return self.__class__(__raw_tensor__=result)
                else:
                    return result

            return func
        return attr

    def __setattr__(self, key, value):
        if key == "requires_grad":
            self.__raw_tensor__.stop_gradient = not value
            return
        Tensor.__setattr__(self, key, value)
