from mf import base
from mf.base import variable_functions
from mf.base.variable_functions import *

nn = base.nn
optimizer = base.optimizer
data = base.data
Tensor = base.Tensor

current_framework = None


def import_torch():
    global Tensor, nn, optimizer, data
    from mf.ports.torch_port.tensor_impl import TensorImpl as Tensor
    from mf.ports.torch_port import nn as nn
    from mf.ports.torch_port import optimizer as optimizer
    from mf.ports.torch_port import data as data
    from mf.ports.torch_port import variable_functions_impl
    import_variable_functions(variable_functions_impl)


def import_paddle():
    import warnings
    # 不知道这个warning目的是什么，暂时忽略。
    warnings.filterwarnings("ignore", message="When training, we now always track global mean and variance.*")

    global Tensor, nn, optimizer, data
    from mf.ports.paddle_port.tensor_impl import TensorImpl as Tensor
    from mf.ports.paddle_port import nn as nn
    from mf.ports.paddle_port import optimizer as optimizer
    from mf.ports.paddle_port import data as data
    from mf.ports.paddle_port import variable_functions_impl
    import_variable_functions(variable_functions_impl)


def config(framework=None):
    global current_framework
    if framework is None:
        try:
            import_torch()
            current_framework = "torch"
        except ImportError:
            try:
                import_paddle()
                current_framework = "paddle"
            except ImportError:
                raise Exception("Cannot find any supported framework")
        return
    if framework == current_framework:
        return
    if framework == 'paddle':
        import_paddle()
    elif framework == 'torch':
        import_torch()
    else:
        raise Exception('Cannot find framework {}'.format(framework))


__all__ = ['Tensor', 'nn', 'optimizer', 'data']


def import_variable_functions(functions):
    for name in dir(functions):
        if name.startswith('__'):
            continue
        globals()[name] = getattr(functions, name)
        __all__.append(name)


import_variable_functions(variable_functions)
