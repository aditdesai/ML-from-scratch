'''
PCA captures the directions with highest variance
principal components - eigenvectors of the covariance matrix

intuitive ex - Imagine a cloud of data points shaped like a stretched ellipse. PCA finds the long axis of the ellipse 
(the direction in which the data is most spread out) as the first principal component. The short axis, perpendicular to 
the long axis, is the second principal component. By rotating the data so that the long axis aligns with the x-axis and 
the short axis aligns with the y-axis, PCA reorients the data in a way that highlights its intrinsic structure.
'''

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # subtract mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, this function needs samples as columns
        cov = np.cov(X.T)

        # eigenvectors and eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # eigenvector transpose
        eigenvectors = eigenvectors.T

        # sort as per eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        X = X - self.mean

        return np.dot(X, self.components.T)