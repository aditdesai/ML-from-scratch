import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self):
        pass

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X: np.ndarray, y: np.ndarray, lr: float=0.01, epochs: int=1000) -> None:
        m, n = X.shape

        self.w = np.random.rand(n, 1) - 0.5
        self.b = 0

        for _ in range(epochs):
            y_pred = self(X)

            dw = np.dot(X.T, y_pred - y) / m
            db = np.sum(y_pred - y) / m

            self.w -= lr * dw
            self.b -= lr * db

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(np.dot(X, self.w) + self.b)


X, y = load_breast_cancer(return_X_y=True)
print(X.shape, y.shape) # (569, 30) and (569,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1, 1)

model = LogisticRegression()
model.train(X_train, y_train)

y_pred = model(X_test)
y_pred = [0 if pred[0] < 0.5 else 1 for pred in y_pred]
accuracy = np.sum(y_test == y_pred) * 100 / len(y_test)
print(f"Accuracy: {accuracy}%")