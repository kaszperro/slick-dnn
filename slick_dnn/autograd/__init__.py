from abc import ABC, abstractmethod

import numpy as np

from slick_dnn import variable


class Context:
    """
    This class is for storing information for back propagation.
    Autograd uses this class instead of using self to allow constructions::

        relu = ReLU()

        b = relu(a)
        c = relu(b)

    That means, that you can use one instance of Autograd class
    to all of your operations. Without it, the above example would be::

        relu1 = ReLU()
        relu2 = ReLU()

        b = relu1(a)
        c = relu2(b)
    """

    def __init__(self):
        self.data_for_back = None

    def save_for_back(self, *data):
        """
        Saves given data for back propagation.

        :param data: Iterable of any data to save.
        :type data: Any
        """
        if len(data) == 1:
            data = data[0]
        self.data_for_back = data


class Autograd(ABC):
    """
    Autograd is a base class for all operations made on Variables.
    """

    def apply(self, *variables_list):
        """
        Actual creation of new Variable.
        It calls overwritten :py:class:`forward` method, creates new :py:class:`Context`
        (same context is used in forward an backward pass)
        and sets :code:`backward_function` and :code:`backward_variables`
        for the new Variable

        :param variables_list: any variables, backward function will have to calculate gradients w.r.t all input variables
        :return: one variable, the one with tracked history and calculated data
        """
        ctx = Context()

        input_tensors = [v.data for v in variables_list]
        forward_tensor = self.forward(ctx, *input_tensors)

        # we set has_grad=False, for performance improvement
        output_variable = variable.Variable(forward_tensor, has_grad=False)
        output_variable.backward_function = lambda x: self.backward(ctx, x)
        output_variable.backward_variables = [v for v in variables_list]

        return output_variable

    @abstractmethod
    def forward(self, ctx: Context, *tensors_list):
        """
        Forward pass of variable. Each Autograd object must implement it.

        :param ctx: Context, classes can save information in them
        :type ctx: Context

        :param tensors_list: any list of input tensors
        :return: one new tensor
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, ctx: Context, grad: np.array = None):
        """
        Backward pass. Each Autograd Object must implement it.

        :param ctx: Same context as in forward pass
        :type ctx: Context

        :param grad: gradient

        :return: gradient w.r.t all inputs
        """
        raise NotImplementedError

    def __call__(self, *variables_list):
        """
        For convenience. One can use all
        Autograd objects by simply calling them
        instead of using :py:class:`apply` method.

        :param variables_list: same as in :py:class:`forward` method
        :return: what :py:class:`apply` returns
        """
        return self.apply(*variables_list)
