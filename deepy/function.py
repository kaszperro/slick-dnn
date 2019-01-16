from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from deepy.variable import Variable


class Context:
    pass


class Function(ABC):

    def apply(self, *variables_list):
        new_context = Context()

        input_tensors = [v.data for v in variables_list]
        forward_tensor = self.forward(new_context, *input_tensors)

        output_variable = Variable(forward_tensor)
        output_variable.backward_function = lambda x: self.backward(new_context, x)
        output_variable.backward_variables = [v for v in variables_list]

        return output_variable

    @staticmethod
    def forward(ctx, *tensors_list):
        pass

    @staticmethod
    def backward(ctx, grad: np.array):
        pass

    def __call__(self, *variables_list):
        return self.apply(*variables_list)


class Add(Function):
    @staticmethod
    def forward(ctx, tensor1: np.array, tensor2: np.array):
        return tensor1 + tensor2

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


class Sub(Function):
    @staticmethod
    def forward(ctx, tensor1: np.array, tensor2: np.array):
        return tensor1 - tensor2

    @staticmethod
    def backward(ctx, grad):
        return grad, -grad


class MatMul(Function):

    @staticmethod
    def forward(ctx, tensor1: np.array, tensor2: np.array):
        ctx.t1 = tensor1
        ctx.t2 = tensor2

        return tensor1 @ tensor2

    @staticmethod
    def backward(ctx, grad: np.array):
        return grad @ np.swapaxes(ctx.t2, -1, -2), np.swapaxes(ctx.t1, -1, -2) @ grad


class ReLU(Function):
    @staticmethod
    def forward(ctx, tensor: np.array):
        ctx.t = tensor
        return np.clip(tensor, a_min=0, a_max=None)

    @staticmethod
    def backward(ctx, grad: np.array):
        print("licze back, zapisany: {}".format(ctx.t))

        new_grad = deepcopy(grad)
        new_grad[ctx.t < 0] = 0
        return new_grad
