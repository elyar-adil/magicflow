import torch

import mf.common.tensor_wrapping as tw
from mf.ports.torch_port.tensor_impl import TensorImpl

no_grad = torch.no_grad


def concat(tensors, dim):
    return tw.wrap_tensor(torch.concat(tw.unwrap_tensor_list(tensors), dim=dim), tensor_wrapper_class=TensorImpl)


def relu(x):
    return tw.wrap_tensor(torch.relu(tw.unwrap_tensor(x)), tensor_wrapper_class=TensorImpl)


def softmax(x, dim=None):
    return tw.wrap_tensor(torch.softmax(tw.unwrap_tensor(x), dim=dim), tensor_wrapper_class=TensorImpl)
