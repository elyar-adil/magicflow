
def unwrap_tensor_list(tensors):
    unwrapped_tensors = []
    for tensor in tensors:
        unwrapped_tensors.append(tensor.__raw_tensor__)
    return unwrapped_tensors


def wrap_tensor(tensor, tensor_wrapper_class):
    return tensor_wrapper_class(__raw_tensor__=tensor)


def unwrap_tensor(tensor):
    return tensor.__raw_tensor__