from deepy.tensor import Tensor
from deepy.tensor import zeros as zero_tensor
from deepy.variable import Variable
from copy import deepcopy
import numpy as np


class Function:

    @staticmethod
    def forward(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Add(Function):
    @staticmethod
    def forward(variable1: Variable, variable2: Variable):
        out_variable = Variable(variable1.shape, variable1.dtype)
        out_variable.tensor = variable1.tensor + variable2.tensor
        out_variable.back_variables = [
            (variable1, lambda x: x),
            (variable2, lambda x: x)

        ]
        return out_variable


class Sub(Function):
    @staticmethod
    def forward(variable1: Variable, variable2: Variable):
        out_variable = Variable(variable1.shape, variable1.dtype)
        out_variable.tensor = variable1.tensor - variable2.tensor
        out_variable.back_variables = [
            (variable1, lambda x: x),
            (variable2, lambda x: -x)
        ]
        return out_variable


class MatMul(Function):

    def simple_matmul(self, t1, t2):
        return t1 @ t2

    @staticmethod
    def forward(variable1: Variable, variable2: Variable):
        out_tensor = variable1.tensor @ variable2.tensor

        out_variable = Variable(out_tensor.shape, out_tensor.dtype)
        out_variable.tensor = out_tensor

        out_variable.back_variables = [
            (variable1, lambda x: x @ variable2.tensor.transpose()),
            (variable2, lambda x: variable1.tensor.transpose() @ x)
        ]

        return out_variable
