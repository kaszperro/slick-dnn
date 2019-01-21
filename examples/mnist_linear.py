import numpy as np

from deepy.autograd.activations import Softmax, Sigmoid
from deepy.autograd.losses import CrossEntropyLoss, MSELoss
from deepy.autograd.optimizers import SGD
from deepy.data import DataLoader
from deepy.data.example_datasets import MNISTTrainDataSet, MNISTTestDataSet
from deepy.module import Linear, Sequential
from deepy.variable import Variable

batch_size = 64
iterations = 15
learning_rate = 0.01

my_model = Sequential(
    Linear(28 * 28, 512),
    Sigmoid(),
    Linear(512, 10),
    Softmax()
)

train_dataset = MNISTTrainDataSet(flatten_input=True, one_hot_output=True)
test_dataset = MNISTTestDataSet(flatten_input=True, one_hot_output=True)

train_data_loader = DataLoader(train_dataset)
test_data_loader = DataLoader(test_dataset)

train_batches = train_data_loader.get_batch_iterable(batch_size)
all_test_batches = test_data_loader.get_all_batches(shuffle=False)
all_test_batches_in = np.array([v[0] for v in all_test_batches]) / 255
all_test_batches_out = [v[1] for v in all_test_batches]

optimizer = SGD(my_model.get_variables_list(), learning_rate)

loss = MSELoss()#CrossEntropyLoss()


def test_model_acc():
    test_input = Variable(all_test_batches_in)
    test_output = my_model(test_input).data

    return np.sum(np.argmax(test_output, axis=1) == np.argmax(all_test_batches_out, axis=1)) / len(all_test_batches_in)


for it in range(iterations):
    train_batches.shuffle()

    for i_b, (batch_in, batch_out) in enumerate(train_batches):
        batch_in = batch_in / 255

        model_input = Variable(batch_in)

        good_output = Variable(batch_out)
        model_output = my_model(model_input)

        err = loss(good_output, model_output)
        print(err.data.shape)
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

    print("iteration: {}, acc: {}".format(it, test_model_acc()))
