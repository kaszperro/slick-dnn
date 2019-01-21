import numpy as np

from deepy.autograd import Autograd


class MSELoss(Autograd):
    @staticmethod
    def forward(ctx, truth, predictions) -> np.array:
        if truth.shape != predictions.shape:
            raise ValueError("Wrong shapes")

        ctx.truth = truth
        ctx.predictions = predictions
        return ((truth - predictions) ** 2).mean()

    @staticmethod
    def backward(ctx, grad: np.array = None):
        num_batches = 1
        if len(ctx.truth.shape) > 1:
            num_batches = ctx.truth.shape[0]

        return 2 / num_batches * (ctx.truth - ctx.predictions), 2 / num_batches * (ctx.predictions - ctx.truth)


class CrossEntropyLoss(Autograd):

    @staticmethod
    def forward(ctx, truth, predictions) -> np.array:
        ctx.truth = truth
        ctx.predictions = predictions
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return - truth * np.log(predictions) - (1 - truth) * np.log(1 - predictions)

    @staticmethod
    def backward(ctx, grad: np.array = None):
        num_batches = 1
        if len(ctx.truth.shape) > 1:
            num_batches = ctx.truth.shape[0]
        p = ctx.predictions
        p = np.clip(p, 1e-15, 1 - 1e-15)

        return (np.log(1 - p) - np.log(p)) / num_batches, (- (ctx.truth / p) + (1 - ctx.truth) / (1 - p)) / num_batches
