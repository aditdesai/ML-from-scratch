import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def deriv_mse(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Add a small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) 
    
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def deriv_binary_cross_entropy(y_true, y_pred):
    return y_pred - y_true

def cross_entropy(y_true, y_pred):
    # Adding small value (epsilon) to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)

    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def deriv_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)

    return -(y_true - y_pred) / y_true.shape[0] # dL/dz - since we're directly calculating derivative wrt the logits, we don't need deriv_softmax
    # return -(y_true / y_pred) / y_true.shape[0] # dL/ds