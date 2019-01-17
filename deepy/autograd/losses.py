import numpy as np

from deepy.autograd import Autograd


class MSELoss(Autograd):
    @staticmethod
    def forward(ctx, truth, predictions) -> np.array:
        if truth.shape != predictions.shape:
            raise ValueError("Wrong shapes")
        if len(truth.shape) == 1:
            ax = 0
        else:
            ax = 1
        ctx.truth = truth
        ctx.predictions = predictions
        return ((truth - predictions) ** 2).mean(axis=ax)

    @staticmethod
    def backward(ctx, grad: np.array = None):
        num_batches = 1
        if len(ctx.truth.shape) > 1:
            num_batches = ctx.truth.shape[0]

        l1 = 2 / num_batches * (ctx.truth - ctx.predictions)
        l2 = 2 / num_batches * (ctx.predictions - ctx.truth)

        return 2 / num_batches * (ctx.truth - ctx.predictions), 2 / num_batches * (ctx.predictions - ctx.truth)
