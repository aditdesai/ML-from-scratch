import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import heapq

class KNearestNeighbours:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _l2norm(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict(self, x):
        dist = [self._l2norm(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(dist)[:self.k]

        k_nearest_labels = y_train[k_indices]

        freq = {}
        for label in k_nearest_labels:
            if label not in freq:
                freq[label] = 1
            else:
                freq[label] += 1

        return max(freq, key=lambda label: freq[label])


    def __call__(self, X):
        return np.array([self._predict(x) for x in X])


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


model = KNearestNeighbours(5)
model.fit(X_train, y_train)

predictions = model(X_test)
print(f"Accuracy: {accuracy(y_test, predictions)}")
