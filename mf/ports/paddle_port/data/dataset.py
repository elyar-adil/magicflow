import abc

from mf.base.data.dataset import Dataset as DatasetBase


class Dataset(DatasetBase, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
