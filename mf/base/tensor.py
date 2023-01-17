import abc
import inspect


class Tensor(abc.ABC):
    def __init__(self, tensor):
        self.__raw_tensor__ = tensor

    def __getattr__(self, item):
        if item == "__raw_tensor__":
            return self.__getattribute__("__raw_tensor__")
        attr = self.__raw_tensor__.__getattribute__(item)
        # TODO: max method returns values and indices in pytorch
        #       but only returns values in paddle
        if inspect.isbuiltin(attr) or inspect.ismethod(attr):
            def func(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, self.__raw_tensor__.__class__):
                    return self.__class__(__raw_tensor__=result)
                else:
                    return result
            return func
        return attr

    def __setattr__(self, key, value):
        if key == "__raw_tensor__":
            self.__dict__["__raw_tensor__"] = value
            return
        self.__raw_tensor__.__setattr__(key, value)

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__class__(__raw_tensor__=self.__raw_tensor__ + other)

    def __radd__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__class__(__raw_tensor__=other + self.__raw_tensor__)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__class__(__raw_tensor__=self.__raw_tensor__ - other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__class__(__raw_tensor__=self.__raw_tensor__ * other)

    def __rmul__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__class__(__raw_tensor__=other * self.__raw_tensor__)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__class__(__raw_tensor__=self.__raw_tensor__ / other)

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__class__(__raw_tensor__=self.__raw_tensor__ @ other)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__class__(__raw_tensor__=self.__raw_tensor__ ** other)

    def __neg__(self):
        return self.__class__(__raw_tensor__=-self.__raw_tensor__)

    def __abs__(self):
        return self.__class__(__raw_tensor__=self.__raw_tensor__.abs())

    def __getitem__(self, item):
        return self.__class__(self.__raw_tensor__[item])

    def __setitem__(self, key, value):
        self.__raw_tensor__[key] = value.__raw_tensor__

    def __len__(self):
        return len(self.__raw_tensor__)

    def __repr__(self):
        return repr(self.__raw_tensor__)

    def __str__(self):
        return str(self.__raw_tensor__)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__raw_tensor__ == other

    def __ne__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__raw_tensor__ != other

    def __lt__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__raw_tensor__ < other

    def __le__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__raw_tensor__ <= other

    def __gt__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__raw_tensor__ > other

    def __ge__(self, other):
        if isinstance(other, Tensor):
            other = other.__raw_tensor__
        return self.__raw_tensor__ >= other

    def __hash__(self):
        return hash(self.__raw_tensor__)

    def __bool__(self):
        return bool(self.__raw_tensor__)

    def __int__(self):
        return int(self.__raw_tensor__)

    def __float__(self):
        return float(self.__raw_tensor__)

    def __complex__(self):
        return complex(self.__raw_tensor__)

    def __index__(self):
        return self.__raw_tensor__.__index__()

    def __round__(self, n=None):
        return round(self.__raw_tensor__, n)

    def __floor__(self):
        return self.__raw_tensor__.__floor__()

    def __ceil__(self):
        return self.__raw_tensor__.__ceil__()

    def __trunc__(self):
        return self.__raw_tensor__.__trunc__()
