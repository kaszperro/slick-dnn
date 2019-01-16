from abc import ABC, abstractmethod

import numpy as np

from deepy.variable import Variable


class Function(ABC):
    def apply(self, *variables_list):
        input_tensors = [v.tensor for v in variables_list]
        forward_tensor = self.forward(*input_tensors)

        output_variable = Variable(forward_tensor)
        output_variable.backward_function = self.backward
        output_variable.backward_variables = [v for v in variables_list]

        return output_variable

    @abstractmethod
    def forward(self, *tensors_list):
        pass

    @abstractmethod
    def backward(self, grad: np.array):
        pass

    def __call__(self, *variables_list):
        return self.apply(*variables_list)


class Add(Function):

    def forward(self, tensor1: np.array, tensor2: np.array):
        return tensor1 + tensor2

    def backward(self, grad):
        return grad, grad


class Sub(Function):

    def forward(self, tensor1: np.array, tensor2: np.array):
        return tensor1 - tensor2

    def backward(self, grad):
        return grad, -grad


class MatMul(Function):

    def __init__(self):
        self.t1 = None
        self.t2 = None

    def forward(self, tensor1: np.array, tensor2: np.array):
        self.t1 = tensor1
        self.t2 = tensor2
        return tensor1 @ tensor2

    def backward(self, grad: np.array):
        return grad @ np.swapaxes(self.t2, -1, -2), np.swapaxes(self.t1, -1, -2) @ grad
