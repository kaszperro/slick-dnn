import deepy as dp
import numpy as np

if __name__ == '__main__':
    a = dp.variable.ones((2, 2))
    b = dp.variable.ones((2, 3))
    c = dp.variable.ones((2, 3))

    b.tensor.holder = np.array([[1, 2, 3], [4, 5, 6]])
    d = a @ b + c
    d.backward(dp.tensor.ones((2, 3)))
    print(a.get_grad())
    print(b.get_grad())
    print(c.get_grad())
