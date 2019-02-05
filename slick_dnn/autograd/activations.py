import numpy as np

from slick_dnn.autograd import Autograd, Context


class ArcTan(Autograd):
    """ Applies the arctan function element-wise. """

    def forward(self, ctx: Context, tensor: np.array):
        """ ArcTan(x) = :math:`tan^{-1}(x)` """

        ctx.save_for_back(tensor)
        return np.arctan(tensor)

    def backward(self, ctx: Context, grad: np.array):
        """ ArcTan(x)' = :math:`\\frac{1}{x^2 + 1}`"""

        tensor = ctx.data_for_back
        return grad / (tensor * tensor + 1)


class ReLU(Autograd):
    """ Applies the ReLU function element-wise. """

    def forward(self, ctx: Context, tensor: np.array):
        """ ReLU(x) = :math:`max(0,x)`"""

        ctx.save_for_back(tensor)
        return np.clip(tensor, a_min=0, a_max=None)

    def backward(self, ctx: Context, grad: np.array):
        """ReLU(x)' = :math:`\\begin{cases}
        0 & \\text{if  }  x < 0 \\newline
        1 & \\text{if  }  x > 0
        \\end{cases}`
        """

        t = ctx.data_for_back
        return np.where(t < 0, 0, grad)


class Sigmoid(Autograd):
    """ Applies the Sigmoid function element-wise. """

    def forward(self, ctx: Context, tensor: np.array) -> np.array:
        """Sigmoid(x) = :math:`\\frac{1}{ 1 + e^{-x} }`"""

        sig = 1 / (1 + np.exp(-tensor))
        ctx.save_for_back(sig)
        return sig

    def backward(self, ctx: Context, grad: np.array = None):
        """Sigmoid(x)' = :math:`\\frac{ e^{-x} }{ (1+ e^{-x})^2 }`"""

        sig = ctx.data_for_back
        return sig * (1 - sig) * grad


class Softmax(Autograd):
    """Applies the Softmax function element-wise. """

    def forward(self, ctx: Context, x: np.array) -> np.array:
        """ :math:`Softmax(x_i) = \\frac{exp(x_i)}{\\sum_j{exp(x_j)}}` """

        e_x = np.exp(x)
        res = e_x / np.sum(e_x, axis=-1, keepdims=True)
        ctx.save_for_back(res)
        return res

    def backward(self, ctx: Context, grad: np.array = None):
        """:math:`Softmax(x_i)' = \\frac{ exp(x_i) * \\sum_{j \\neq i}{exp(x_j)} }{ (\\sum_j{exp(x_j)})^2 }`"""

        res = ctx.data_for_back
        return grad * res * (1 - res)


class Softplus(Autograd):
    """ Applies the softplus function element-wise. """

    def forward(self, ctx: Context, tensor: np.array) -> np.array:
        """ Softplus(x) = :math:`ln(1 + e^x)` """

        ctx.save_for_back(1 + np.exp(-tensor))
        return np.log(1 + np.exp(tensor))

    def backward(self, ctx: Context, grad: np.array = None):
        """ Softplus'(x) = :math:`\\frac{1}{1 + e^{-x}}` """

        denominator = ctx.data_for_back
        return grad / denominator


class Softsign(Autograd):
    """ Applies the softsign function element-wise. """

    def forward(self, ctx: Context, tensor: np.array) -> np.array:
        """ Softsign(x) = :math:`\\frac{1}{1 + |x|}` """

        denominator = 1 + np.abs(tensor)
        ctx.save_for_back(denominator)
        return tensor / denominator

    def backward(self, ctx: Context, grad: np.array = None):
        """ Softsign'(x) = :math:`\\frac{1}{(1 + |x|)^2}` """

        denominator = ctx.data_for_back
        return grad / (denominator * denominator)


class Tanh(Autograd):
    """ Applies the tanh function element-wise. """

    def forward(self, ctx: Context, tensor: np.array) -> np.array:
        """Tanh(x)"""

        tanh = np.tanh(tensor)
        ctx.save_for_back(tanh)
        return tanh

    def backward(self, ctx: Context, grad: np.array = None):
        """ Tanh(x)' = :math:`1 - Tanh^2(x)` """

        tanh = ctx.data_for_back
        return (1 - tanh * tanh) * grad
