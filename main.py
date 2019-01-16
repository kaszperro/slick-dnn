import deepy as dp
from deepy.function import ReLU
from deepy.variable import Variable
import numpy as np

if __name__ == '__main__':
    a = Variable(np.ones((2, 2)))
    b = Variable(np.zeros((2, 2)))
    my_rel = ReLU()

    c = my_rel(a + b)
    d = my_rel(c + a)

    d.backward(np.ones((2, 2)))
