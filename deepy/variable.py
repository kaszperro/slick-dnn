import numpy as np

from deepy.tensor import Tensor
from deepy.tensor import zeros as tens_zero
from deepy.tensor import ones as tens_ones


class Variable:
    def __init__(self, shape, dtype=np.float32, has_grad=True):
        self.shape = shape
        self.has_grad = has_grad
        if has_grad:
            self.grad_tensor = tens_zero(shape, dtype)

        self.tensor = Tensor(shape, dtype)
        self.back_variables = []
        self.dtype = dtype

    def get_tensor(self):
        return self.tensor

    def get_grad(self):
        return self.grad_tensor

    def __getitem__(self, item):
        return self.tensor.__getitem__(item)

    def __setitem__(self, key, value):
        self.tensor.__setitem__(key, value)

    def backward(self, acc: Tensor):
        self.grad_tensor += acc

        for (bv, bf) in self.back_variables:
            bv.backward(bf(acc))

    # mathematical operations

    def __add__(self, other):
        from deepy.autograd.function import Add
        return Add()(self, other)

    def __sub__(self, other):
        from deepy.autograd.function import Sub
        return Sub()(self, other)

    def __matmul__(self, other):
        from deepy.autograd.function import MatMul
        return MatMul()(self, other)


def zeros(shape, dtype=np.float32):
    ret_var = Variable(shape, dtype)
    ret_var.tensor = tens_zero(shape, dtype)
    return ret_var


def ones(shape, dtype=np.float32):
    ret_var = Variable(shape, dtype)
    ret_var.tensor = tens_ones(shape, dtype)
    return ret_var
