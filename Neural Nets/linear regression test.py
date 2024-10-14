from nn import NeuralNet
from layers import FCLayer
from loss import mse, deriv_mse
import numpy as np

# training data of form y = 0.7*x + 0.3
w, b = 0.7, 0.3
x_train = np.expand_dims(np.arange(0, 1, 0.02), axis=1)
y_train = w * x_train + b

print(x_train.shape, y_train.shape)

net = NeuralNet()
net.add(FCLayer(1, 1))

net.use(mse, deriv_mse)
net.train_with_adagrad(x_train, y_train, epochs=200, lr=0.1, batch_size=1)

print(net.layers[0].weights, net.layers[0].bias)

test_input = np.array([[1]])
print(net(test_input))