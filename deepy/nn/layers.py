from abc import ABC, abstractmethod

from deepy.tensor import Tensor
from deepy.variable import Variable


class Layer(ABC):
    @abstractmethod
    def forward(self, inputs: Variable):
        pass

    @abstractmethod
    def backward(self, grad: Variable) -> Variable:
        pass

    def __call__(self, inputs: Variable) -> Variable:
        return self.forward(inputs)


class Dense(Layer):
    def __init__(self, input_futures, output_futures):
        self.input_futures = input_futures
        self.output_futures = output_futures

        self.weights_matrix = Variable((self.input_futures, self.output_futures))
        self.bias_matrix = Variable(self.output_futures)

    def forward(self, inputs: Tensor) -> Tensor:
        return

    def backward(self):
        pass
