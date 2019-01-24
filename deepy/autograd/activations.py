import numpy as np

from deepy.autograd import Autograd


class ArcTan(Autograd):
    """ Applies the arctan function element-wise. """
    @staticmethod
    def forward(ctx, tensor: np.array):
        ctx.t = tensor
        return np.arctan(tensor)

    @staticmethod
    def backward(ctx, grad: np.array):
        return grad / (ctx.t * ctx.t + 1)


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


class Softplus(Autograd):
    """ Applies the softplus function element-wise:

    Softplus(x) = ln(1 + e^x)
    Softplus'(x) = 1 / (1 + e^-x)
    """
    @staticmethod
    def forward(ctx, tensor: np.array) -> np.array:
        ctx.denom = 1 + np.exp(-tensor)
        return np.log(1 + np.exp(tensor))

    @staticmethod
    def backward(ctx, grad: np.array = None):
        return grad / ctx.denom


class Softsign(Autograd):
    """ Applies the softsign function element-wise:


    Softsign(x) = 1 / (1 + |x|)
    Softsign'(x) = 1 / (1 + |x|)^2
    """
    @staticmethod
    def forward(ctx, tensor: np.array) -> np.array:
        ctx.denom = 1 + np.abs(tensor)
        return tensor / ctx.denom

    @staticmethod
    def backward(ctx, grad: np.array = None):
        return grad / (ctx.denom * ctx.denom)


class Tanh(Autograd):
    """ Applies the tanh function element-wise. """
    @staticmethod
    def forward(ctx, tensor: np.array) -> np.array:
        ctx.tanh = np.tanh(tensor)
        return ctx.tanh

    @staticmethod
    def backward(ctx, grad: np.array = None):
        return (1 - ctx.tanh * ctx.tanh) * grad
