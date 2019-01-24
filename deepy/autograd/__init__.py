from abc import ABC, abstractmethod

import numpy as np

from deepy.variable import Variable


class Autograd(ABC):
    class Context:
        """
        This class is for storing information for backpropagation.
        Create this class instead of using self to allow constructions:

        relu = ReLU()

        b = relu(a)
        c = relu(b)

        That means, that you can use one instance of Autograd class
        to all of your operations. Without it, the above example would be:

        relu1 = ReLU()
        relu2 = ReLU()

        b =  relu1(a)
        c = relu2(b)
        """

        def __init__(self):
            self.data_for_back = None

        def save_for_back(self, *tensors):
            if len(tensors) == 1:
                tensors = tensors[0]
            self.data_for_back = tensors

    def apply(self, *variables_list):
        ctx = Autograd.Context()

        input_tensors = [v.data for v in variables_list]
        forward_tensor = self.forward(ctx, *input_tensors)

        output_variable = Variable(forward_tensor)
        output_variable.backward_function = lambda x: self.backward(ctx, x)
        output_variable.backward_variables = [v for v in variables_list]

        return output_variable

    def forward(self, ctx: Context, *tensors_list):
        raise NotImplementedError

    def backward(self, ctx: Context, grad: np.array = None):
        raise NotImplementedError

    def __call__(self, *variables_list):
        return self.apply(*variables_list)
