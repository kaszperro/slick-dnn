from abc import ABC, abstractmethod

import numpy as np


class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class DataLoader:
    class SingleIterable:
        def __init__(self, dataset: Dataset):
            self.dataset = dataset

            self.indexes = list(range(len(self.dataset)))

        def shuffle(self):
            np.random.shuffle(self.indexes)

        def __len__(self):
            return self.dataset.__len__()

        def __getitem__(self, idx):
            return self.dataset[self.indexes[idx]]

    class BatchIterable:
        def __init__(self, dataset: Dataset, batch_size: int):
            self.dataset = dataset
            self.batch_size = batch_size
            self.indexes = list(range(len(self.dataset)))

        def shuffle(self):
            np.random.shuffle(self.indexes)

        def __len__(self):
            return self.dataset.__len__() // self.batch_size

        def __getitem__(self, idx):
            if idx >= len(self):
                raise IndexError("index out of range")
            return self.dataset[self.indexes[idx:idx + self.batch_size]]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def get_single_iterable(self) -> SingleIterable:
        return self.SingleIterable(self.dataset)

    def get_batch_iterable(self, batch_size) -> BatchIterable:
        return self.BatchIterable(self.dataset, batch_size)

    def get_all_batches(self, shuffle=False):
        batches = [v for v in self.dataset]
        if shuffle:
            np.random.shuffle(batches)
        return np.array(batches)
