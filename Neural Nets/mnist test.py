from nn import NeuralNet
from layers import FCLayer, ActivationLayer
from activations import softmax, deriv_softmax, leaky_relu, deriv_leaky_relu, tanh, deriv_tanh, relu, deriv_relu
from loss import cross_entropy, deriv_cross_entropy
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255.

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255.

y_train = to_categorical(y_train)

print(x_train.shape, y_train.shape)

net = NeuralNet()
net.add(FCLayer(28*28, 512))
net.add(ActivationLayer(relu, deriv_relu))
net.add(FCLayer(512, 10))
net.add(ActivationLayer(softmax, deriv_softmax))

net.use(cross_entropy, deriv_cross_entropy)
net.train_with_gd(x_train[:20000], y_train[:20000], epochs=50, lr=0.1, batch_size=50)


predictions = net(x_test)
predicted_classes = [np.argmax(prediction) for prediction in predictions]
accuracy = np.sum(np.array(predicted_classes) == np.array(y_test)) / len(y_test)

print(f"Accuracy on test set: {accuracy * 100:.2f}%")