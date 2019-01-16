import deepy as dp
from deepy.variable import Variable
import numpy as np

if __name__ == '__main__':
    a = Variable([[1, 2], [3, 4]])
    b = Variable(np.ones((2, 2)))
    c = Variable(np.ones((2, 2)))
    d = a @ b + c
    d.backward(np.ones((2, 2)))

    print(a.grad)
    print(b.grad)
