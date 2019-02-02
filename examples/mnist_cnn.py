import numpy as np

from deepy.autograd.tensor_modifications import MaxPool2d
from deepy.module import Conv2d
from deepy.variable import Variable

input_image = np.array([
    [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
        ]
    ],
    [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
        ]
    ]
])

input_var = Variable(input_image)

my_conv = Conv2d(2, 1, (1, 1))

after_conf = my_conv(input_var)

after_conf.backward(np.ones_like(after_conf.data))

max_pool = MaxPool2d(2, 1)

max_out = max_pool(after_conf)

max_out.backward(np.ones_like(max_out.data))


print(my_conv.weights.grad)
