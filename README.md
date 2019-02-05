# Slick-dnn

Deep learning library written in python just for fun. 

It uses numpy for computations. API is similar to PyTorch's one.

###Docs:

https://slick-dnn.readthedocs.io/en/latest/

### Includes:

1. Activation functions:
    * ArcTan
    * ReLU
    * Sigmoid
    * Softmax
    * Softplus
    * Softsign
    * Tanh
    
2. Losses:
    * MSE
    * Cross Entropy

3. Optimizers:
    * SGD
    * Adam

4. Layers:
    * Linear
    * Conv2d
    * Sequential
    
5. Autograd operations:
    * Reshape
    * Flatten
    * SwapAxes
    * Img2Col
    * MaxPool2d
    * AvgPool2d
    * MatMul
    * Mul
    * Sub
    * Add

### Examples:

* In examples directory there is a MNIST linear classifier, which scores over 96% accuracy on test set.

* In examples directory there is also MNIST CNN classifier, which scored 99.19% accuracy on test set. One epoch of training takes about 290 seconds. It took 7 epochs to reach 99.19% accuracy (~30 min). Time measured on i5-4670k

* Sequential model creation:
```python
from slick_dnn.module import Linear, Sequential
from slick_dnn.autograd.activations import Softmax, ReLU
my_model = Sequential(
    Linear(28 * 28, 300),
    ReLU(),
    Linear(300, 300),
    ReLU(),
    Linear(300, 10),
    Softmax()
    )
```
* Losses:
```python
from slick_dnn.module import Linear
from slick_dnn.autograd.losses import CrossEntropyLoss, MSELoss
from slick_dnn.variable import Variable
import numpy as np

my_model = Linear(10, 10)

loss1 = CrossEntropyLoss()
loss2 = MSELoss()


good_output = Variable(np.zeros((10,10)))
model_input = Variable(np.ones((10,10)))
model_output = my_model(model_input)

error = loss1(good_output, model_output)

# now you can propagate error backwards:
error.backward()
```

* Optimizers:

```python
from slick_dnn.module import Linear
from slick_dnn.autograd.losses import CrossEntropyLoss, MSELoss
from slick_dnn.variable import Variable
from slick_dnn.autograd.optimizers import SGD
import numpy as np


my_model = Linear(10, 10)

loss1 = CrossEntropyLoss()
loss2 = MSELoss()

optimizer1 = SGD(my_model.get_variables_list())

good_output = Variable(np.zeros((10,10)))
model_input = Variable(np.ones((10,10)))
model_output = my_model(model_input)

error = loss1(good_output, model_output)

# now you can propagate error backwards:
error.backward()

# and then optimizer can update variables:
optimizer1.zero_grad()
optimizer1.step()

```

