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
        print("licze back, zapisany: {}".format(ctx.t))

        new_grad = deepcopy(grad)
        new_grad[ctx.t < 0] = 0
        return new_grad


class Sigmoid(Autograd):
    @staticmethod
    def forward(ctx, tensor: np.array) -> np.array:
        ctx.sig = 1 / (1 + np.exp(-tensor))
        return ctx.sig

    @staticmethod
    def backward(ctx, grad: np.array = None):
        return ctx.sig*(1-ctx.sig)*grad
