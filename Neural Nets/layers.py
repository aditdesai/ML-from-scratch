import numpy as np
from typing import Callable
from scipy import signal

# Abstract class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Computes layer output y for given input X
    def forward_prop(self, input: np.ndarray) -> np.ndarray:
        pass
    
    # Computes dL/dW and dL/db for a given output_error dL/dy. Returns dL/dX, dL/dW and dL/db
    def backward_prop(self, dy: np.ndarray) -> np.ndarray:
        pass
    


class FCLayer(Layer):
    def __init__(self, input_neurons: int, output_neurons: int):
        # He initialization
        self.weights = np.random.normal(0, np.sqrt(2. / input_neurons), size=(input_neurons, output_neurons))
        self.bias = np.zeros((1, output_neurons))

    def forward_prop(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 1:
            input = input.reshape(1, -1)

        # input shape - (1, input_neurons)
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    
    def backward_prop(self, dy: np.ndarray) -> np.ndarray:
        if dy.ndim == 1:
            dy = dy.reshape(1, -1)

        # dy shape - (1, output_neurons)
        dX = np.dot(dy, self.weights.T)
        self.dW = np.dot(self.input.T, dy)
        self.db = dy

        return dX
    


class ActivationLayer(Layer):
    def __init__(self, activation: Callable, deriv_activation: Callable):
        self.activation = activation
        self.deriv_activation = deriv_activation

    def forward_prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = self.activation(self.input)

        return self.output
    
    def backward_prop(self, dy: np.ndarray) -> np.ndarray:
        dX = self.deriv_activation(self.input) * dy

        return dX
    

class Conv2dLayer(Layer):
    def __init__(self, input_shape: tuple, num_kernels: int, kernel_size: int):
        self.input_shape = input_shape
        self.num_kernels = num_kernels

        self.output_shape = (num_kernels, input_shape[2] - kernel_size + 1, input_shape[1] - kernel_size + 1) # d x h x w
        self.kernel_shape = (num_kernels, input_shape[0], kernel_size, kernel_size)

        self.weights = np.random.randn(*self.kernel_shape)
        self.bias = np.random.randn(*self.output_shape)

    def forward_prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = np.copy(self.bias)

        for i in range(self.num_kernels):
            for j in range(self.input_shape[0]):
                self.output[i] += signal.correlate2d(self.input[j], self.weights[i, j], "valid")

        return self.output
    
    def backward_prop(self, dy: np.ndarray) -> np.ndarray:
        dK = np.zeros(self.kernel_shape)
        dX = np.zeros(self.input_shape)

        for i in range(self.num_kernels):
            for j in range(self.input_shape[0]):
                dK[i, j] = signal.correlate2d(self.input[j], dy[i], "valid")
                dX[j] += signal.convolve2d(dy[i], self.weights[i, j], "full")

        self.dW = dK
        self.db = dy

        return dX
    

class FlattenLayer(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_prop(self, input: np.ndarray) -> np.ndarray:
        return np.reshape(input, self.output_shape)
    
    def backward_prop(self, dy: np.ndarray) -> np.ndarray:
        return np.reshape(dy, self.input_shape)