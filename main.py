import numpy as np

from deepy.autograd.activations import Sigmoid, ReLU
from deepy.autograd.losses import MSELoss
from deepy.autograd.optimizers import SGD
from deepy.module import Linear
from deepy.variable import Variable

if __name__ == '__main__':
    lin1 = Linear(40, 3)
    sig = Sigmoid()
    rel = ReLU()
    lin2 = Linear(3, 3)

    numpy_input = np.ones(40)
    numpy_output = np.ones(3)

    input_var = Variable(numpy_input)
    output_var = Variable(numpy_output)

    loss = MSELoss()

    optimizer = SGD([*lin1.variables_list, *lin2.variables_list], learning_rate=0.01)
    for i in range(100):
        model_out = lin2(rel(lin1(input_var)))  # lin2(lin1(input_var))
        err = loss(model_out, output_var)
        print(err.data)
        optimizer.zero_grad()
        err.backward()
        optimizer.step()
        # print(lin2.weights.grad)

    print(
        lin2(rel(lin1(input_var)))
    )
