from nn import NeuralNet
from layers import FCLayer, ActivationLayer
from activations import relu, deriv_relu, leaky_relu, deriv_leaky_relu
from loss import mse, deriv_mse
import numpy as np

# training data of form y = 2*x1 + 3*x2 + 1
w = np.array([2, 3])
b = 1

x_train = np.zeros((100, 2))
for i in range(100):
    x_train[i] = np.array([np.random.randint(0, 10), np.random.randint(0, 10)])

y_train = np.dot(x_train, w) + b
y_train = y_train.reshape(-1, 1)

print(x_train.shape, y_train.shape)

net = NeuralNet()
net.add(FCLayer(2, 5))
net.add(ActivationLayer(leaky_relu, deriv_leaky_relu))
net.add(FCLayer(5, 1))

net.use(mse, deriv_mse)
net.train_with_rmsprop(x_train, y_train, epochs=1000, lr=0.01, batch_size=1)

test_input = np.array([[1, 5]])
print(test_input.shape)
print(net(test_input))