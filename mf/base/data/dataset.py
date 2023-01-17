import abc


class Dataset(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def __getitem__(self, item):
        pass
