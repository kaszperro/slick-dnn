import numpy as np


class Variable:
    def __init__(self, from_numpy: np.array, has_grad=True):
        self.tensor = np.array(from_numpy)

        self.has_grad = has_grad
        if has_grad:
            self.grad = np.zeros_like(self.tensor, dtype=np.float32)

        self.backward_function = None
        self.backward_variables = []

    def __getitem__(self, item):
        return self.tensor.__getitem__(item)

    def __setitem__(self, key, value):
        self.tensor.__setitem__(key, value)

    def backward(self, grad: np.array):
        self.grad += grad

        if len(self.backward_variables) > 1:
            accumulated = self.backward_function(grad)
            for i, bv in enumerate(self.backward_variables):
                bv.backward(accumulated[i])
        elif len(self.backward_variables) == 1:
            accumulated = self.backward_function(grad)
            self.backward_variables[0].backward(accumulated)

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
