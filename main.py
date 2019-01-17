import deepy as dp
from deepy.autograd.losses import MSELoss
from deepy.autograd.optimizers import SGD
from deepy.module import Linear
from deepy.variable import Variable
import numpy as np

if __name__ == '__main__':
    lin1 = Linear(40, 10)
    numpy_input = np.ones((3, 40))
    input_var = Variable(numpy_input)
    good_out = Variable(3*np.ones((3, 10)))

    loss = MSELoss()

    optimizer = SGD(lin1.variables_list)

    for i in range(100):
        model_out = lin1(input_var)
        err = loss(model_out, good_out)

        optimizer.zero_grad()
        err.backward()
        optimizer.step()

    print(lin1(Variable(np.ones(40))))
