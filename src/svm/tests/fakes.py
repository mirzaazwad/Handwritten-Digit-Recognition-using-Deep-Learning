import numpy as np

class FakeSVM:
    def __init__(self, *_, **__):
        self._trained = False

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._trained = True
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)