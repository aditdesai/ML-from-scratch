import math
import numpy as np

class NeuralNet:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.deriv_loss = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, deriv_loss):
        self.loss = loss
        self.deriv_loss = deriv_loss

    def train_with_sgd(self, X, y, epochs=10, lr=0.01, batch_size=32):
        num_training_examples = len(X)
        num_batches = math.ceil(num_training_examples / batch_size)

        for i in range(epochs):
            err = 0
            for j in range(num_batches):
                start_idx = j * batch_size
                end_idx =  min((j + 1) * batch_size, num_training_examples)

                batch_X = X[start_idx : end_idx]
                batch_y = y[start_idx: end_idx]

                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer.dW = np.zeros_like(layer.weights)
                        layer.db = np.zeros_like(layer.bias)

                for k in range(len(batch_X)):

                    # Forward Propagation
                    output = batch_X[k]
                    for layer in self.layers:
                        output = layer.forward_prop(output)

                    err += self.loss(batch_y[k], output)

                    # Backward Propagation
                    dX = self.deriv_loss(batch_y[k], output)
                    for layer in reversed(self.layers):
                        dX = layer.backward_prop(dX)
                
                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer.dW /= len(batch_X)
                        layer.db /= len(batch_X)

                for layer in self.layers:     
                    if hasattr(layer, 'weights'):   
                        layer.weights = layer.weights - lr * layer.dW
                        layer.bias = layer.bias - lr * layer.db

            err /= num_training_examples
            print(f"Epoch: {i} / {epochs}, Loss: {err}")

    def train_with_adagrad(self, X, y, epochs=10, lr=0.01, batch_size=32, epsilon=1e-8):
        num_training_examples = len(X)
        num_batches = math.ceil(num_training_examples / batch_size)

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                # Moment (velocity)
                layer.v_w = np.zeros_like(layer.weights)
                layer.v_b = np.zeros_like(layer.bias)

        for i in range(epochs):
            err = 0
            for j in range(num_batches):
                start_idx = j * batch_size
                end_idx =  min((j + 1) * batch_size, num_training_examples)

                batch_X = X[start_idx : end_idx]
                batch_y = y[start_idx: end_idx]

                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer.dW = np.zeros_like(layer.weights)
                        layer.db = np.zeros_like(layer.bias)

                for k in range(len(batch_X)):

                    # Forward Propagation
                    output = batch_X[k]
                    for layer in self.layers:
                        output = layer.forward_prop(output)

                    err += self.loss(batch_y[k], output)

                    # Backward Propagation
                    dX = self.deriv_loss(batch_y[k], output)
                    for layer in reversed(self.layers):
                        dX = layer.backward_prop(dX)
                
                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer.dW /= len(batch_X)
                        layer.db /= len(batch_X)

                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        
                        layer.v_w = layer.v_w + np.square(layer.dW)
                        layer.v_b = layer.v_b + np.square(layer.db)

                        layer.weights -= lr * layer.dW / (np.sqrt(layer.v_w) + epsilon)
                        layer.bias -= lr * layer.db / (np.sqrt(layer.v_b) + epsilon)

            err /= num_training_examples
            print(f"Epoch: {i} / {epochs}, Loss: {err}")

    def train_with_rmsprop(self, X, y, epochs=10, lr=0.01, p=0.9, batch_size=32, epsilon=1e-8):
        num_training_examples = len(X)
        num_batches = math.ceil(num_training_examples / batch_size)

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                # Moment (velocity)
                layer.v_w = np.zeros_like(layer.weights)
                layer.v_b = np.zeros_like(layer.bias)

        for i in range(epochs):
            err = 0
            for j in range(num_batches):
                start_idx = j * batch_size
                end_idx =  min((j + 1) * batch_size, num_training_examples)

                batch_X = X[start_idx : end_idx]
                batch_y = y[start_idx: end_idx]

                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer.dW = np.zeros_like(layer.weights)
                        layer.db = np.zeros_like(layer.bias)

                for k in range(len(batch_X)):

                    # Forward Propagation
                    output = batch_X[k]
                    for layer in self.layers:
                        output = layer.forward_prop(output)

                    err += self.loss(batch_y[k], output)

                    # Backward Propagation
                    dX = self.deriv_loss(batch_y[k], output)
                    for layer in reversed(self.layers):
                        dX = layer.backward_prop(dX)
                
                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer.dW /= len(batch_X)
                        layer.db /= len(batch_X)

                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        
                        layer.v_w = p * layer.v_w + (1-p) * np.square(layer.dW)
                        layer.v_b = p * layer.v_b + (1-p) * np.square(layer.db)

                        layer.weights -= lr * layer.dW / (np.sqrt(layer.v_w) + epsilon)
                        layer.bias -= lr * layer.db / (np.sqrt(layer.v_b) + epsilon)

            err /= num_training_examples
            print(f"Epoch: {i} / {epochs}, Loss: {err}")

    def train_with_adam(self, X, y, p1=0.9, p2=0.999, epochs=10, lr=0.01, batch_size=32, epsilon=1e-8):
        num_training_examples = len(X)
        num_batches = math.ceil(num_training_examples / batch_size)

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                # First moment (momentum)
                layer.m_w = np.zeros_like(layer.weights)
                layer.m_b = np.zeros_like(layer.bias)
                # Second moment (velocity)
                layer.v_w = np.zeros_like(layer.weights)
                layer.v_b = np.zeros_like(layer.bias)

        t = 0
        for i in range(epochs):
            err = 0
            for j in range(num_batches):
                start_idx = j * batch_size
                end_idx =  min((j + 1) * batch_size, num_training_examples)

                batch_X = X[start_idx : end_idx]
                batch_y = y[start_idx: end_idx]

                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer.dW = np.zeros_like(layer.weights)
                        layer.db = np.zeros_like(layer.bias)

                for k in range(len(batch_X)):

                    # Forward Propagation
                    output = batch_X[k]
                    for layer in self.layers:
                        output = layer.forward_prop(output)

                    err += self.loss(batch_y[k], output)

                    # Backward Propagation
                    dX = self.deriv_loss(batch_y[k], output)
                    for layer in reversed(self.layers):
                        dX = layer.backward_prop(dX)
                
                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer.dW /= len(batch_X)
                        layer.db /= len(batch_X)

                t += 1
                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        # Update first moment (momentum)
                        layer.m_w = p1 * layer.m_w + (1 - p1) * layer.dW
                        layer.m_b = p1 * layer.m_b + (1 - p1) * layer.db
                        
                        # Update second moment (velocity)
                        layer.v_w = p2 * layer.v_w + (1 - p2) * np.square(layer.dW)
                        layer.v_b = p2 * layer.v_b + (1 - p2) * np.square(layer.db)

                        # Compute bias corrections
                        m_w_hat = layer.m_w / (1 - p1**t)
                        m_b_hat = layer.m_b / (1 - p1**t)
                        v_w_hat = layer.v_w / (1 - p2**t)
                        v_b_hat = layer.v_b / (1 - p2**t)

                        # Update parameters
                        layer.weights -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                        layer.bias -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

            err /= num_training_examples
            print(f"Epoch: {i} / {epochs}, Loss: {err}")


    def __call__(self, input):
        num_training_examples = len(input)

        result = []
        for i in range(num_training_examples):
            output = input[i]

            for layer in self.layers:
                output = layer.forward_prop(output)

            result.append(output)

        return result
