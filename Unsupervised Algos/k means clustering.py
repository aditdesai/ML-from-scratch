import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k, max_epochs=100):
        self.k = k
        self.epochs = max_epochs

    def _assign_labels(self, X):
        labels = np.zeros(X.shape[0], dtype=np.int32)

        for i in range(X.shape[0]):
            min_dist = float('inf')

            for j in range(self.k):
                # Euclidean distance from each centroid
                dist = np.sqrt(np.sum((X[i] - self.centroids[j]) ** 2))

                if dist < min_dist:
                    min_dist = dist
                    labels[i] = j

        return labels


    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.k, X.shape[1]))

        for k in range(self.k):
            # Get all points in this cluster
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[k] = self.centroids[k]

        return new_centroids

    def fit(self, X):
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)] # Generates k numbers from [0, 300)

        for _ in range(self.epochs):
            # Assign each data point to nearest centroid
            labels = self._assign_labels(X)

            # Update centroid
            new_centroids = self._update_centroids(X, labels)

            # Convergence check
            if np.all(np.abs(self.centroids - new_centroids) < 1e-10):
                break
            
            self.centroids = new_centroids

        self._labels = labels


    def plot_clusters(self, X):
        plt.figure(figsize=(10, 6))

        plt.scatter(X[:, 0], X[:, 1], c=self._labels, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')

        plt.title("K-means clustering results")
        plt.legend()
        plt.show()


X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

model = KMeansClustering(k=3)
model.fit(X)

model.plot_clusters(X)