from abc import ABC, abstractmethod

import numpy as np


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
    def __init__(self, variables_list: list, learning_rate=0.01,
                 learning_rate_decay=1., momentum=0.):
        super().__init__(variables_list)
        self.lr = learning_rate
        self.lrd = learning_rate_decay
        self.mu = momentum
        self.vel = [np.zeros_like(v) for v in self.variables_list]

    def step(self):
        for i, v in enumerate(self.variables_list):
            self.vel[i] = self.mu * self.vel[i] - self.lr * v.grad
            v.data += self.vel[i]

        self.lr *= self.lrd
