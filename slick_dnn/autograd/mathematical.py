import numpy as np

from slick_dnn.autograd import Autograd


class Add(Autograd):
    def forward(self, ctx, tensor1: np.array, tensor2: np.array):
        return tensor1 + tensor2

    def backward(self, ctx, grad):
        return grad, grad


class Sub(Autograd):
    def forward(self, ctx, tensor1: np.array, tensor2: np.array):
        return tensor1 - tensor2

    def backward(self, ctx, grad):
        return grad, -grad


class MatMul(Autograd):
    def forward(self, ctx, tensor1: np.array, tensor2: np.array):
        ctx.save_for_back(tensor1, tensor2)
        return tensor1 @ tensor2

    def backward(self, ctx, grad: np.array):
        t1, t2 = ctx.data_for_back

        grad_1d = len(grad.shape) == 1
        t1_1d = len(t1.shape) == 1

        grad = np.atleast_2d(grad)
        t1 = np.atleast_2d(t1)
        t2 = np.atleast_2d(t2)

        grad1 = grad @ np.swapaxes(t2, -1, -2)
        grad2 = np.swapaxes(t1, -1, -2) @ grad

        if grad_1d:
            grad1 = np.squeeze(grad1)

        if t1_1d:
            grad2 = np.squeeze(grad2)

        return grad1, grad2


class Mul(Autograd):
    def forward(self, ctx, tensor1: np.array, tensor2: np.array):
        ctx.save_for_back(tensor1, tensor2)
        return tensor1 * tensor2

    def backward(self, ctx, grad: np.array):
        t1, t2 = ctx.data_for_back
        return grad * t2, grad * t1
