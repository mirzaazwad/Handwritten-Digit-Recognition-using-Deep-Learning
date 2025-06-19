import numpy as np
import pytest

class DummyMNIST:
    def __init__(self, *_):
        self._X = np.arange(10 * 784).reshape(10, 784).astype(np.uint8)
        self._y = np.arange(10)

    def load_training(self):
        return self._X, self._y

    def load_testing(self):
        return self._X, self._y
    
class FakeRFC:
    def __init__(self, *_, **__):
        self._trained = False

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._trained = True
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


@pytest.fixture
def tiny_mnist(monkeypatch):
    import src.rfc.rfc as rfc_mod
    monkeypatch.setattr(rfc_mod,"MNIST", DummyMNIST)

@pytest.fixture
def dummy_rfc(monkeypatch):
    import src.rfc.rfc as rfc_mod
    monkeypatch.setattr(rfc_mod,"RandomForestClassifier", FakeRFC)

@pytest.fixture(autouse=True)
def mute_matplotlib(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *_, **__: None)
