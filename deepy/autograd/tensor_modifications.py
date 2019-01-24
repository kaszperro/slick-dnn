import numpy as np

from deepy.autograd import Autograd


class Reshape(Autograd):
    def __init__(self, *new_shape):
        self.new_shape = new_shape

    def forward(self, ctx: Autograd.Context, tensor: np.array):
        ctx.save_for_back(tensor.shape)
        return np.reshape(tensor, self.new_shape)

    def backward(self, ctx: Autograd.Context, grad: np.array):
        old_shape = ctx.data_for_back
        return np.reshape(grad, old_shape)


class GetItem(Autograd):
    def __init__(self, item):
        self.item = item

    def forward(self, ctx: Autograd.Context, tensor):
        ctx.save_for_back(tensor.shape)
        return tensor[self.item]

    def backward(self, ctx: Autograd.Context, grad: np.array):
        old_shape = ctx.data_for_back
        new_grad = np.zeros(old_shape, dtype=np.float32)
        new_grad[self.item] = grad
        return new_grad
