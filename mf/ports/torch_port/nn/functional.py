import torch.nn.functional as torch_function

import mf.common.tensor_wrapping as tw
from mf.ports.torch_port.tensor_impl import TensorImpl


def cross_entropy(input, target):
    return tw.wrap_tensor(torch_function.cross_entropy(tw.unwrap_tensor(input), tw.unwrap_tensor(target)), TensorImpl)
