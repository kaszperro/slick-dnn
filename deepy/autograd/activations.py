import numpy as np

from deepy.autograd import Autograd


class ReLU(Autograd):
    def forward(self, ctx: Autograd.Context, tensor: np.array):
        ctx.save_for_back(tensor)
        return np.clip(tensor, a_min=0, a_max=None)

    def backward(self, ctx: Autograd.Context, grad: np.array):
        t = ctx.data_for_back
        return np.where(t < 0, 0, grad)


class Sigmoid(Autograd):
    def forward(self, ctx: Autograd.Context, tensor: np.array) -> np.array:
        sig = 1 / (1 + np.exp(-tensor))
        ctx.save_for_back(sig)
        return sig

    def backward(self, ctx: Autograd.Context, grad: np.array = None):
        sig = ctx.data_for_back
        return sig * (1 - sig) * grad


class Softmax(Autograd):
    def forward(self, ctx: Autograd.Context, x: np.array) -> np.array:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        res = e_x / np.sum(e_x, axis=-1, keepdims=True)
        ctx.save_for_back(res)
        return res

    def backward(self, ctx: Autograd.Context, grad: np.array = None):
        res = ctx.data_for_back
        return grad * res * (1 - res)
