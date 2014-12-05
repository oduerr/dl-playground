# Taken from https://gist.github.com/duschendestroyer/5170087
import numpy as np
from scipy import linalg
from sklearn.utils import array2d, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator


class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10**-3, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        X = X.T
        examples = np.shape(X)[1]
        sigma = np.dot(X,X.T) / (examples - 1)
        U, S, V = linalg.svd(sigma)
        d = np.sqrt(1/S[0:100])
        dd = np.append(d, np.zeros((np.shape(X)[0] - 100)))
        #tmp = np.dot(U, np.diag(1/np.sqrt(S +self.regularization)))
        tmp = np.dot(U, np.diag(dd))
        self.components_ = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X = as_float_array(X)
        X_transformed = X - self.mean_
        return np.dot(X_transformed, self.components_)