import numpy as np

from deepy.autograd.mathematical import Add
from deepy.autograd.mathematical import MatMul
from deepy.autograd.mathematical import Mul
from deepy.autograd.mathematical import Sub
from deepy.autograd.tensor_modifications import Reshape
from deepy.autograd.tensor_modifications import SwapAxes


class Variable:
    def __init__(self, from_numpy: np.array, has_grad=True):
        self.data = np.array(from_numpy)

        self.has_grad = has_grad
        self.grad = None
        if has_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

        self.backward_function = None
        self.backward_variables = []

        self.shape = self.data.shape

    def backward(self, grad: np.array = None):
        if grad is not None:
            self.grad = grad + self.grad

        if self.backward_function is not None:
            accumulated = self.backward_function(grad)
            for i, bv in enumerate(self.backward_variables):
                bv.backward(
                    accumulated[i]
                    if len(self.backward_variables) > 1
                    else accumulated
                )

    def __str__(self):
        return "<Variable>\n" + self.data.__str__()

    def __getitem__(self, item):
        from deepy.autograd.tensor_modifications import GetItem
        return GetItem(item)(self)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    # mathematical operations

    def reshape(self, *new_shape):
        return Reshape(*new_shape)(self)

    def swap_axes(self, axis1, axis2):
        return SwapAxes(axis1, axis2)(self)

    def __add__(self, other):
        return Add()(self, other)

    def __sub__(self, other):
        return Sub()(self, other)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def __mul__(self, other):
        return Mul()(self, other)


def zeros(shape, dtype=np.float32) -> Variable:
    return Variable(np.zeros(shape, dtype=dtype))


def ones(shape, dtype=np.float32) -> Variable:
    return Variable(np.ones(shape, dtype=dtype))
