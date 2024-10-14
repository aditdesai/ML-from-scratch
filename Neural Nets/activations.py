import numpy as np

def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

def deriv_leaky_relu(x):
    return np.where(x > 0, 1, 0.01)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    '''
    s = sigmoid(x)
    return s * (1 - s)
    '''
    return 1 # This implementation assumes sigmoid is only used in the output layer with the binary cross entropy loss

def tanh(x):
    return np.tanh(x)

def deriv_tanh(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stability improvement by subtracting max
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def deriv_softmax(x):
    return 1 # since the deriv_cross_entropy() function is directly returning dL/dz


if __name__ == "__main__":
    print(relu(np.array([0, 5, 9, -10, -5, 3, -2, -6, 7, 2, -3])))
    print(deriv_relu(np.array([0, 5, 9, -10, -5, 3, -2, -6, 7, 2, -3])))