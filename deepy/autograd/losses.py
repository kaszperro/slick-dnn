import numpy as np

from deepy.autograd import Autograd


class MSELoss(Autograd):

    def forward(self, ctx, truth, predictions) -> np.array:
        if truth.shape != predictions.shape:
            raise ValueError("Wrong shapes")

        ctx.save_for_back(truth, predictions)
        return ((truth - predictions) ** 2).mean()

    def backward(self, ctx, grad: np.array = None):
        truth, predictions = ctx.data_for_back

        num_batches = 1
        if len(truth.shape) > 1:
            num_batches = truth.shape[0]

        return 2 / num_batches * (truth - predictions), 2 / num_batches * (predictions - truth)


class CrossEntropyLoss(Autograd):
    def forward(self, ctx, truth, predictions) -> np.array:
        ctx.save_for_back(truth, predictions)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return - truth * np.log(predictions) - (1 - truth) * np.log(1 - predictions)

    def backward(self, ctx, grad: np.array = None):
        truth, predictions = ctx.data_for_back

        num_batches = 1
        if len(truth.shape) > 1:
            num_batches = truth.shape[0]
        p = predictions
        p = np.clip(p, 1e-15, 1 - 1e-15)

        return (np.log(1 - p) - np.log(p)) / num_batches, (- (truth / p) + (1 - truth) / (1 - p)) / num_batches
