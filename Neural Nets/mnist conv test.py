from nn import NeuralNet
from layers import FCLayer, ActivationLayer, Conv2dLayer, FlattenLayer
from activations import softmax, deriv_softmax, relu, deriv_relu
from loss import cross_entropy, deriv_cross_entropy
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_train = y_train.reshape(len(y_train), 1, 10)

print(x_train.shape, y_train.shape)

net = NeuralNet()
net.add(Conv2dLayer((1, 28, 28), 5, 3))
net.add(ActivationLayer(relu, deriv_relu))
net.add(Conv2dLayer((5, 26, 26), 7, 3))
net.add(ActivationLayer(relu, deriv_relu))
net.add(FlattenLayer((7, 24, 24), (1, 7*24*24)))
net.add(FCLayer(7*24*24, 10))
net.add(ActivationLayer(softmax, deriv_softmax))

net.use(cross_entropy, deriv_cross_entropy)
net.train_with_adam(x_train[:1000], y_train[:1000], epochs=75, lr=0.01, batch_size=8)


predictions = net(x_test)
predicted_classes = [np.argmax(prediction) for prediction in predictions]
accuracy = np.sum(np.array(predicted_classes) == np.array(y_test)) / len(y_test)

print(f"Accuracy on test set: {accuracy * 100:.2f}%")