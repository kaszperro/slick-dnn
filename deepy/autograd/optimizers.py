from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, variables_list: list):
        self.variables_list = variables_list

    def zero_grad(self):
        for v in self.variables_list:
            v.grad.fill(0)

    @abstractmethod
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, variables_list: list, learning_rate=0.01):
        super().__init__(variables_list)
        self.lr = learning_rate

    def step(self):
        for v in self.variables_list:
            v.data -= self.lr * v.grad
