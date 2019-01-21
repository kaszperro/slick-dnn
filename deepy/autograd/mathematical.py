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
        grad_1d = len(grad.shape) == 1
        t1_1d = len(ctx.t1.shape) == 1

        grad = np.atleast_2d(grad)
        ctx.t1 = np.atleast_2d(ctx.t1)
        ctx.t2 = np.atleast_2d(ctx.t2)

        grad1 = grad @ np.swapaxes(ctx.t2, -1, -2)
        grad2 = np.swapaxes(ctx.t1, -1, -2) @ grad

        if grad_1d:
            grad1 = np.squeeze(grad1)

        if t1_1d:
            grad2 = np.squeeze(grad2)

        return grad1, grad2
