import numpy as np


class Tensor:
    def __init__(self, shape, dtype=np.float32):
        self.holder = np.empty(shape, dtype=dtype)
        self.dtype = dtype
        self.shape = shape

    def __getitem__(self, item):
        return self.holder.__getitem__(item)

    def __setitem__(self, key, value):
        self.holder.__setitem__(key, value)

    def __add__(self, other):
        return from_holder(self.holder.__add__(other.holder))

    def __sub__(self, other):
        return from_holder(self.holder.__sub__(other.holder))

    def __neg__(self):
        return from_holder(self.holder.__neg__())

    def __matmul__(self, other):
        return from_holder(self.holder.__matmul__(other.holder))

    def transpose(self):
        return from_holder(np.swapaxes(self.holder, -1, -2))

    def __str__(self):
        return "[Tensor] " + self.holder.__str__()


def from_holder(holder: np.array) -> Tensor:
    my_ret = Tensor(holder.shape, holder.dtype)
    my_ret.holder = holder
    return my_ret


def zeros(shape, dtype=np.float32):
    ret_tens = Tensor(shape, dtype)
    ret_tens.holder = np.zeros(shape, dtype)
    return ret_tens


def ones(shape, dtype=np.float32):
    ret_tens = Tensor(shape, dtype)
    ret_tens.holder = np.ones(shape, dtype)
    return ret_tens
