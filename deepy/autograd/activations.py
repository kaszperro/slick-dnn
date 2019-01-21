from copy import deepcopy

import numpy as np

from deepy.autograd import Autograd


class ReLU(Autograd):
    @staticmethod
    def forward(ctx, tensor: np.array):
        ctx.t = tensor
        return np.clip(tensor, a_min=0, a_max=None)

    @staticmethod
    def backward(ctx, grad: np.array):
        new_grad = deepcopy(grad)
        new_grad[ctx.t <= 0] = 0

        return new_grad


class Sigmoid(Autograd):
    @staticmethod
    def forward(ctx, tensor: np.array) -> np.array:
        ctx.sig = 1 / (1 + np.exp(-tensor))
        return ctx.sig

    @staticmethod
    def backward(ctx, grad: np.array = None):
        return ctx.sig * (1 - ctx.sig) * grad


class Softmax(Autograd):

    @staticmethod
    def forward(ctx, x: np.array) -> np.array:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        ctx.res = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return ctx.res

    @staticmethod
    def backward(ctx, grad: np.array = None):
        return grad * ctx.res * (1 - ctx.res)
