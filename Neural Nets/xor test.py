from nn import NeuralNet
from layers import FCLayer, ActivationLayer
from activations import tanh, deriv_tanh
from loss import mse, deriv_mse
import numpy as np

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = NeuralNet()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, deriv_tanh))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, deriv_tanh))

net.use(mse, deriv_mse)
net.train_with_rmsprop(x_train, y_train, epochs=1000, lr=0.1, batch_size=1)

print(net(x_train))