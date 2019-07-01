import numpy as np

from slick_dnn.autograd import Autograd


class Add(Autograd):
    """Adds given tensors"""

    def forward(self, ctx, tensor1, tensor2):
        return tensor1 + tensor2

    def backward(self, ctx, grad):
        return grad, grad


class Sub(Autograd):
    """ Subtracts given tensors: tensor1-tensor2"""

    def forward(self, ctx, tensor1, tensor2):
        return tensor1 - tensor2

    def backward(self, ctx, grad):
        return grad, -grad


class MatMul(Autograd):
    """Matrix multiplication: tensor1 @ tensor2"""

    def forward(self, ctx, tensor1, tensor2):
        ctx.save_for_back(tensor1, tensor2)
        return tensor1 @ tensor2

    def backward(self, ctx, grad: np.array):
        t1, t2 = ctx.data_for_back

        grad1 = grad @ np.swapaxes(t2, -1, -2)
        grad2 = np.swapaxes(t1, -1, -2) @ grad

        return grad1, grad2


class Mul(Autograd):
    """Element-wise multiplication"""

    def forward(self, ctx, tensor1, tensor2):
        ctx.save_for_back(tensor1, tensor2)
        return tensor1 * tensor2

    def backward(self, ctx, grad: np.array):
        t1, t2 = ctx.data_for_back
        return grad * t2, grad * t1
