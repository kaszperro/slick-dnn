from abc import ABC, abstractmethod

import numpy as np

from deepy.variable import Variable


class Module(ABC):
    def __init__(self):
        self.variables_list = []

    def register_variable(self, var: Variable):
        self.variables_list.append(var)

    def register_variables(self, *var_iterable):
        self.variables_list.extend(var_iterable)

    def get_variables_list(self) -> list:
        return self.variables_list

    @abstractmethod
    def forward(self, *input_variables) -> Variable:
        raise NotImplementedError

    def __call__(self, *input_variables) -> Variable:
        return self.forward(*input_variables)


class Sequential(Module):
    def __init__(self, *sequences):
        super().__init__()

        self.sequences_list = list(sequences)
        for seq in self.sequences_list:
            try:
                var_list = seq.get_variables_list()

                self.register_variables(*var_list)
            except AttributeError:
                pass

    def forward(self, *input_variables):
        out = self.sequences_list[0](*input_variables)
        for seq in self.sequences_list[1:]:
            out = seq(out)
        return out


class Linear(Module):
    def __init__(self, num_input, num_output):
        super().__init__()

        self.num_input = num_input
        self.num_output = num_output

        self.weights = Variable(np.random.normal(0, 0.05, (self.num_input, self.num_output)))
        self.biases = Variable(np.zeros(self.num_output, dtype=np.float32))

        self.register_variables(self.weights, self.biases)

    def forward(self, in_var):
        return (in_var @ self.weights) + self.biases
