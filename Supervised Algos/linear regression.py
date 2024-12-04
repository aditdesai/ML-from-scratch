import numpy as np

class LinearRegression:
    def __init__(self):
        pass

    def train(self, X: np.ndarray, y: np.ndarray, lr: float=0.01, epochs: int=1000) -> None:
        m, n = X.shape

        self.w = np.random.rand(n, 1) - 0.5
        self.b = 0

        for _ in range(epochs):
            y_pred = self(X).reshape(-1, 1)

            dw = np.dot(X.T, y_pred - y) / m
            db = np.sum(y_pred - y) / m

            self.w -= lr * dw
            self.b -= lr * db

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b


# training data of form y = 2*x1 + 3*x2 + 1
w = np.array([2, 3])
b = 1

x_train = np.zeros((100, 2))
for i in range(100):
    x_train[i] = np.array([np.random.randint(0, 10), np.random.randint(0, 10)])

y_train = np.dot(x_train, w) + b
y_train = y_train.reshape(-1, 1)

model = LinearRegression()
model.train(x_train, y_train)

print(model.w, model.b) # w1 should be 2, w2 should b2 3 and b should be 1

test_input = np.array([[1, 5], [3, 8], [0.5, 1]])
print(model(test_input)) # output should be 18, 30 and 4