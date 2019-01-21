import numpy as np

from deepy.autograd.activations import Softmax, Sigmoid, ReLU
from deepy.autograd.losses import CrossEntropyLoss, MSELoss
from deepy.autograd.optimizers import SGD
from deepy.data import DataLoader
from deepy.data.example_datasets import MNISTTrainDataSet, MNISTTestDataSet
from deepy.module import Linear, Sequential
from deepy.variable import Variable

batch_size = 64
iterations = 10
learning_rate = 0.005

my_model = Sequential(
    Linear(28 * 28, 300),
    ReLU(),
    Linear(300, 300),
    ReLU(),
    Linear(300, 10),
    Softmax()
)

train_dataset = MNISTTrainDataSet(flatten_input=True, one_hot_output=True, input_normalization=(0.1307, 0.3081))
test_dataset = MNISTTestDataSet(flatten_input=True, one_hot_output=True, input_normalization=(0.1307, 0.3081))

train_data_loader = DataLoader(train_dataset)
test_data_loader = DataLoader(test_dataset)

train_batches = train_data_loader.get_batch_iterable(batch_size)
test_batches = test_data_loader.get_batch_iterable(batch_size)

optimizer = SGD(my_model.get_variables_list(), learning_rate)

loss = CrossEntropyLoss()


def test_model_acc():
    correct = 0
    for test_batch_in, test_batch_out in test_batches:
        test_output = my_model(Variable(test_batch_in)).data
        correct += np.sum(np.argmax(test_output, axis=1) == np.argmax(test_batch_out, axis=1))

    return correct / len(test_dataset)


finished = False
for it in range(iterations):
    if finished:
        break
    train_batches.shuffle()

    for i_b, (batch_in, batch_out) in enumerate(train_batches):
        model_input = Variable(batch_in)

        good_output = Variable(batch_out)
        model_output = my_model(model_input)

        err = loss(good_output, model_output)
        optimizer.zero_grad()
        err.backward()

        optimizer.step()

        if i_b % 50 == 0:
            acc = test_model_acc()
            print("model accuracy: {}".format(acc))
            if acc > 0.97:
                finished = True
                break
