from paddle import Tensor as PaddleTensor
from paddle.io import DataLoader as PaddleDataLoader
from paddle.io import Dataset as PaddleDataset

from mf.ports.paddle_port.tensor_impl import TensorImpl as Tensor


class DataLoader:

    def __init__(self, dataset, batch_size=1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = False):
        super().__init__()

        self.__dataloader = PaddleDataLoader(DatasetUnwrap(dataset), batch_size=batch_size, shuffle=shuffle,
                                             num_workers=num_workers, use_shared_memory=pin_memory)

    def __iter__(self):
        return Iterator(self.__dataloader.__iter__())


class DatasetUnwrap(PaddleDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data = unwrap_tensor(data)
        return data

    def __len__(self):
        return len(self.dataset)


class Iterator:
    def __init__(self, raw_iterator):
        self.raw_iterator = raw_iterator

    def __next__(self):
        try:
            data = self.raw_iterator.__next__()
            data = wrap_tensor(data)
            return data
        except StopIteration:
            raise StopIteration


def unwrap_tensor(data):
    if isinstance(data, tuple):
        data = tuple([x.__raw_tensor__ for x in data])
    elif isinstance(data, list):
        data = list([x.__raw_tensor__ for x in data])
    elif isinstance(data, Tensor):
        data = data.__raw_tensor__
    else:
        raise NotImplementedError
    return data


def wrap_tensor(data):
    if isinstance(data, tuple):
        data = tuple([Tensor(__raw_tensor__=x) for x in data])
    elif isinstance(data, list):
        data = list([Tensor(__raw_tensor__=x) for x in data])
    elif isinstance(data, PaddleTensor):
        data = Tensor(__raw_tensor__=data)
    else:
        raise NotImplementedError
    return data
