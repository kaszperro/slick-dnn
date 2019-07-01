import numpy as np

from slick_dnn.autograd import Autograd


class ArcTan(Autograd):
    """ Applies the arctan function element-wise. """

    def forward(self, ctx, x):
        """ ArcTan(x) = :math:`tan^{-1}(x)` """

        ctx.save_for_back(x)
        return np.arctan(x)

    def backward(self, ctx, grad):
        """ ArcTan(x)' = :math:`\\frac{1}{x^2 + 1}`"""

        tensor, = ctx.data_for_back
        return grad / (tensor * tensor + 1)


class ReLU(Autograd):
    """ Applies the ReLU function element-wise. """

    def forward(self, ctx, x):
        """ ReLU(x) = :math:`max(0,x)`"""

        ctx.save_for_back(x)
        return np.clip(x, a_min=0, a_max=None)

    def backward(self, ctx, grad):
        """ReLU(x)' = :math:`\\begin{cases}
        0 & \\text{if  }  x < 0 \\newline
        1 & \\text{if  }  x > 0
        \\end{cases}`
        """

        t, = ctx.data_for_back
        return np.where(t < 0, 0, grad)


class Sigmoid(Autograd):
    """ Applies the Sigmoid function element-wise. """

    def forward(self, ctx, x):
        """Sigmoid(x) = :math:`\\frac{1}{ 1 + e^{-x} }`"""

        sig = 1 / (1 + np.exp(-x))
        ctx.save_for_back(sig)
        return sig

    def backward(self, ctx, grad):
        """Sigmoid(x)' = :math:`\\frac{ e^{-x} }{ (1+ e^{-x})^2 }`"""

        sig, = ctx.data_for_back
        return sig * (1 - sig) * grad


class Softmax(Autograd):
    """Applies the Softmax function element-wise. """

    def forward(self, ctx, x) -> np.array:
        """ :math:`Softmax(x_i) = \\frac{exp(x_i)}{\\sum_j{exp(x_j)}}` """

        e_x = np.exp(x)
        res = e_x / np.sum(e_x, axis=-1, keepdims=True)
        ctx.save_for_back(res)
        return res

    def backward(self, ctx, grad):
        """:math:`Softmax(x_i)' = \\frac{ exp(x_i) * \\sum_{j \\neq i}{exp(x_j)} }{ (\\sum_j{exp(x_j)})^2 }`"""

        res, = ctx.data_for_back
        return grad * res * (1 - res)


class Softplus(Autograd):
    """ Applies the softplus function element-wise. """

    def forward(self, ctx, x) -> np.array:
        """ Softplus(x) = :math:`ln(1 + e^x)` """

        ctx.save_for_back(1 + np.exp(-x))
        return np.log(1 + np.exp(x))

    def backward(self, ctx, grad):
        """ Softplus'(x) = :math:`\\frac{1}{1 + e^{-x}}` """

        denominator, = ctx.data_for_back
        return grad / denominator


class Softsign(Autograd):
    """ Applies the softsign function element-wise. """

    def forward(self, ctx, x):
        """ Softsign(x) = :math:`\\frac{1}{1 + |x|}` """

        denominator = 1 + np.abs(x)
        ctx.save_for_back(denominator)
        return x / denominator

    def backward(self, ctx, grad):
        """ Softsign'(x) = :math:`\\frac{1}{(1 + |x|)^2}` """

        denominator, = ctx.data_for_back
        return grad / (denominator * denominator)


class Tanh(Autograd):
    """ Applies the tanh function element-wise. """

    def forward(self, ctx, x):
        """Tanh(x)"""

        tanh = np.tanh(x)
        ctx.save_for_back(tanh)
        return tanh

    def backward(self, ctx, grad):
        """ Tanh(x)' = :math:`1 - Tanh^2(x)` """

        tanh, = ctx.data_for_back
        return (1 - tanh * tanh) * grad
