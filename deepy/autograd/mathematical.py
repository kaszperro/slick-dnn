import numpy as np

from deepy.autograd import Autograd


class Add(Autograd):
    @staticmethod
    def forward(ctx, tensor1: np.array, tensor2: np.array):
        return tensor1 + tensor2

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


class Sub(Autograd):
    @staticmethod
    def forward(ctx, tensor1: np.array, tensor2: np.array):
        return tensor1 - tensor2

    @staticmethod
    def backward(ctx, grad):
        return grad, -grad


class MatMul(Autograd):

    @staticmethod
    def forward(ctx, tensor1: np.array, tensor2: np.array):
        ctx.t1 = tensor1
        ctx.t2 = tensor2

        return tensor1 @ tensor2

    @staticmethod
    def backward(ctx, grad: np.array):
        grad = np.atleast_2d(grad)
        ctx.t1 = np.atleast_2d(ctx.t1)
        ctx.t2 = np.atleast_2d(ctx.t2)

        return grad @ np.swapaxes(ctx.t2, -1, -2), np.swapaxes(ctx.t1, -1, -2) @ grad
