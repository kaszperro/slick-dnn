import numpy as np

from slick_dnn.autograd import Autograd, Context


class ArcTan(Autograd):
    """ Applies the arctan function element-wise. """

    def forward(self, ctx: Context, tensor: np.array):
        ctx.save_for_back(tensor)
        return np.arctan(tensor)

    def backward(self, ctx: Context, grad: np.array):
        tensor = ctx.data_for_back
        return grad / (tensor * tensor + 1)


class ReLU(Autograd):
    def forward(self, ctx: Context, tensor: np.array):
        ctx.save_for_back(tensor)
        return np.clip(tensor, a_min=0, a_max=None)

    def backward(self, ctx: Context, grad: np.array):
        t = ctx.data_for_back
        return np.where(t < 0, 0, grad)


class Sigmoid(Autograd):
    def forward(self, ctx: Context, tensor: np.array) -> np.array:
        sig = 1 / (1 + np.exp(-tensor))
        ctx.save_for_back(sig)
        return sig

    def backward(self, ctx: Context, grad: np.array = None):
        sig = ctx.data_for_back
        return sig * (1 - sig) * grad


class Softmax(Autograd):
    def forward(self, ctx: Context, x: np.array) -> np.array:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        res = e_x / np.sum(e_x, axis=-1, keepdims=True)
        ctx.save_for_back(res)
        return res

    def backward(self, ctx: Context, grad: np.array = None):
        res = ctx.data_for_back
        return grad * res * (1 - res)


class Softplus(Autograd):
    """ Applies the softplus function element-wise:

    Softplus(x) = ln(1 + e^x)
    Softplus'(x) = 1 / (1 + e^-x)
    """

    def forward(self, ctx: Context, tensor: np.array) -> np.array:
        ctx.save_for_back(1 + np.exp(-tensor))
        return np.log(1 + np.exp(tensor))

    def backward(self, ctx: Context, grad: np.array = None):
        denominator = ctx.data_for_back
        return grad / denominator


class Softsign(Autograd):
    """ Applies the softsign function element-wise:


    Softsign(x) = 1 / (1 + |x|)
    Softsign'(x) = 1 / (1 + |x|)^2
    """

    def forward(self, ctx: Context, tensor: np.array) -> np.array:
        denominator = 1 + np.abs(tensor)
        ctx.save_for_back(denominator)
        return tensor / denominator

    def backward(self, ctx: Context, grad: np.array = None):
        denominator = ctx.data_for_back
        return grad / (denominator * denominator)


class Tanh(Autograd):
    """ Applies the tanh function element-wise. """

    def forward(self, ctx: Context, tensor: np.array) -> np.array:
        tanh = np.tanh(tensor)
        ctx.save_for_back(tanh)
        return tanh

    def backward(self, ctx: Context, grad: np.array = None):
        tanh = ctx.data_for_back
        return (1 - tanh * tanh) * grad
