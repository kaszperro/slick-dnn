from abc import ABC, abstractmethod

import numpy as np

from deepy.variable import Variable


class Context:
    pass


class Autograd(ABC):

    @classmethod
    def apply(cls, *variables_list):
        new_context = Context()

        input_tensors = [v.data for v in variables_list]
        forward_tensor = cls.forward(new_context, *input_tensors)

        output_variable = Variable(forward_tensor)
        output_variable.backward_function = lambda x: cls.backward(new_context, x)
        output_variable.backward_variables = [v for v in variables_list]

        return output_variable

    @staticmethod
    @abstractmethod
    def forward(ctx, *tensors_list) -> np.array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(ctx, grad: np.array = None):
        raise NotImplementedError

    def __call__(self, *variables_list):
        return self.apply(*variables_list)
