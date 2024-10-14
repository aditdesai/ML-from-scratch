from nn import NeuralNet
from layers import FCLayer, ActivationLayer
from activations import softmax, deriv_softmax, leaky_relu, deriv_leaky_relu
from loss import cross_entropy, deriv_cross_entropy
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Iris.csv")
dataset = dataset.drop('Id', axis=1)

le = LabelEncoder()
dataset.iloc[:, 4] = le.fit_transform(dataset.iloc[:, 4])

dataset = dataset.sample(frac=1).reset_index(drop=True) # shuffle the dataset

X = dataset.iloc[:, 0:4].to_numpy()
y = dataset.iloc[:, 4].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = to_categorical(y_train)


net = NeuralNet()
net.add(FCLayer(4, 16))
net.add(ActivationLayer(leaky_relu, deriv_leaky_relu))
net.add(FCLayer(16, 3))
net.add(ActivationLayer(softmax, deriv_softmax))

net.use(cross_entropy, deriv_cross_entropy)
net.train_with_sgd(X_train, y_train, epochs=200, lr=0.01, batch_size=1)


predictions = net(X_test)
predicted_classes = [np.argmax(prediction) for prediction in predictions]
accuracy = np.sum(np.array(predicted_classes) == np.array(y_test)) / len(y_test)

print(f"Accuracy on test set: {accuracy * 100:.2f}%")