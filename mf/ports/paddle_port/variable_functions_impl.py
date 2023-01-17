import paddle

import mf.common.tensor_wrapping as tw
from mf.ports.paddle_port.tensor_impl import TensorImpl

no_grad = paddle.no_grad


def concat(tensors, dim):
    return tw.wrap_tensor(paddle.concat(tw.unwrap_tensor_list(tensors), axis=dim), tensor_wrapper_class=TensorImpl)


def relu(x):
    return tw.wrap_tensor(paddle.nn.functional.relu(tw.unwrap_tensor(x)), tensor_wrapper_class=TensorImpl)


def softmax(x, dim=None):
    return tw.wrap_tensor(paddle.nn.functional.softmax(tw.unwrap_tensor(x), axis=dim), tensor_wrapper_class=TensorImpl)
