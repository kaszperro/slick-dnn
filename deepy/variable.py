import numpy as np


class Variable:
    def __init__(self, from_numpy: np.array, has_grad=True):
        self.data = np.array(from_numpy)

        self.has_grad = has_grad
        self.grad = None
        if has_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

        self.backward_function = None
        self.backward_variables = []

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def backward(self, grad: np.array = None):
        if grad is not None:
            if len(grad.shape) - 1 == len(self.grad.shape) or grad.shape[0] != self.grad.shape[0]:
                self.grad += np.sum(grad, 0)
            else:
                self.grad += grad

        if self.backward_function is not None:
            accumulated = self.backward_function(grad)
            for i, bv in enumerate(self.backward_variables):
                bv.backward(
                    accumulated[i]
                    if len(self.backward_variables) > 1
                    else accumulated
                )

    def __str__(self):
        return "[Variable] " + self.data.__str__()

    # mathematical operations

    def reshape(self, *new_shape):
        from deepy.autograd.mathematical import Reshape
        return Reshape(new_shape)(self)

    def __add__(self, other):
        from deepy.autograd.mathematical import Add
        return Add()(self, other)

    def __sub__(self, other):
        from deepy.autograd.mathematical import Sub
        return Sub()(self, other)

    def __matmul__(self, other):
        from deepy.autograd.mathematical import MatMul
        return MatMul()(self, other)

    def __mul__(self, other):
        from deepy.autograd.mathematical import Mul
        return Mul()(self, other)


def zeros(shape, dtype=np.float32) -> Variable:
    return Variable(np.zeros(shape, dtype=dtype))


def ones(shape, dtype=np.float32) -> Variable:
    return Variable(np.ones(shape, dtype=dtype))
