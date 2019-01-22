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
    def __init__(self, variables_list: list, learning_rate=0.01):
        super().__init__(variables_list)
        self.lr = learning_rate

    def step(self):
        for variable in self.variables_list:
            if variable.grad is not None:
                variable.data -= self.lr * variable.grad


class Adam(Optimizer):
    def __init__(self, variables_list: list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(variables_list)

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

        self.state = {}

    @staticmethod
    def initialize_state(state, variable):
        state['step'] = 0
        state['m'] = np.zeros(variable.grad.shape)
        state['v'] = np.zeros(variable.grad.shape)

    def step(self):
        for variable in self.variables_list:
            if variable.grad is None:
                continue

            if variable not in self.state:
                self.state[variable] = {}

            state = self.state[variable]

            # initialisation of state for this variable
            if len(state) == 0:
                self.initialize_state(state, variable)

            state['step'] += 1
            state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * variable.grad
            state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * np.power(variable.grad, 2)

            m_hat = state['m'] / (1 - self.beta1 ** state['step'])
            v_hat = state['v'] / (1 - self.beta2 ** state['step'])

            variable.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
