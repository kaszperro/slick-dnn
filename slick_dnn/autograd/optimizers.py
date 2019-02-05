from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """
    Optimizers receives list of variables to optimize.
    You can call :py:class:`step` to update variables data using their gradient
    """
    def __init__(self, variables_list: list):
        """
        Initialization, sets which Variables to track.

        :param variables_list: which Variables to track and update
        :type variables_list: list

        """
        self.variables_list = variables_list
        self.state = {}

    def zero_grad(self):
        """
        Sets tracked Variables gradient to zero.
        """
        for v in self.variables_list:
            v.grad.fill(0)

    @abstractmethod
    def step(self):
        """
        Each Optimizer has to override this method.
        In step method, actual step of optimization is made.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Momentum in SGD, mimics 'ball' with mass and velocity.
    ::

        state['v'] = self.mu * state['v'] - self.lr * variable.grad

    For each variable, we calculate 'v', as you can see, your 'ball' gathers speed.

    """
    def __init__(self, variables_list: list, learning_rate=0.01, momentum=0., decay=0.):
        """
        Initialization of SGD Optimizer.

        :param variables_list: which Variables to track and update
        :type variables_list: list

        :param learning_rate: learning rate
        :type learning_rate: float

        :param momentum: velocity of momentum
        :type momentum: float

        :param decay: decay of learning rate
        :type decay: float

        """

        super().__init__(variables_list)
        self.lr = learning_rate
        self.decay = decay
        self.mu = momentum

    @staticmethod
    def initialize_state(state, variable):
        # Velocity for momentum.
        state['v'] = np.zeros_like(variable.grad)

    def step(self):
        for variable in self.variables_list:
            if variable.grad is None:
                continue

            if variable not in self.state:
                self.state[variable] = {}

            state = self.state[variable]

            if len(state) == 0:
                self.initialize_state(state, variable)

            state['v'] = self.mu * state['v'] - self.lr * variable.grad
            variable.data += state['v']

        # Linear decay.
        self.lr = self.lr / (1 + self.decay)


class Adam(Optimizer):
    """
    Adam optimizer calculates for each Variable:

    Exponential moving average of gradient and Exponential moving average of squared gradient values.
    """
    def __init__(self, variables_list: list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialization of Adam Optimizer.

        :param variables_list: which Variables to track and update
        :type variables_list: list

        :param learning_rate: learning rate
        :type learning_rate: float

        :param beta1:
        :param beta2:

        :param epsilon: term added to the denominator to improve numerical stability
        :type epsilon: float
        """

        super().__init__(variables_list)

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

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
