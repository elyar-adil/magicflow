import abc


class DataLoader(abc.ABC):
    def __init__(self, dataset, batch_size=1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = False):
        super().__init__()

    @abc.abstractmethod
    def __iter__(self):
        pass
