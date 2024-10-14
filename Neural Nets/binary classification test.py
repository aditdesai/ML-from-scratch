from nn import NeuralNet
from layers import FCLayer, ActivationLayer, Conv2dLayer, FlattenLayer
from activations import sigmoid, deriv_sigmoid, relu, deriv_relu
from loss import binary_cross_entropy, deriv_binary_cross_entropy
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)

    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

net = NeuralNet()
net.add(Conv2dLayer((1, 28, 28), 5, 3))
net.add(ActivationLayer(relu, deriv_relu))
net.add(FlattenLayer((5, 26, 26), (1, 5*26*26)))
net.add(FCLayer(5*26*26, 256))
net.add((ActivationLayer(relu, deriv_relu)))
net.add(FCLayer(256, 2))
net.add(ActivationLayer(sigmoid, deriv_sigmoid))

net.use(binary_cross_entropy, deriv_binary_cross_entropy)
net.train_with_adam(x_train, y_train, epochs=10, lr=0.001, batch_size=8)


predictions = net(x_test)
predicted_classes = [np.argmax(pred[0]) for pred in predictions]
true_classes = np.argmax(y_test, axis=1)

accuracy = np.sum(np.array(predicted_classes) == true_classes) / len(y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")