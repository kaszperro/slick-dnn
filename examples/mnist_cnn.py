import numpy as np

from deepy.module import Conv2d
from deepy.variable import Variable

shape1 = np.ones((2, 4, 100))

b = np.ones(100)

np.add(b, shape1)

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

my_conv = Conv2d(2, 100, (1, 1))

after_conf = my_conv(input_var)
print(after_conf.shape)
