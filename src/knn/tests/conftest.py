import numpy as np
import pytest
from types import ModuleType
import sys
import numpy as np

class FakeKNN:
        def __init__(self, *_, **__):
            self._trained = False

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self._trained = True
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)
        
_fake_mod = ModuleType("fake_knn")
_fake_mod.FakeKNN = FakeKNN
sys.modules["fake_knn"] = _fake_mod
FakeKNN.__module__ = "fake_knn"    

class DummyMNIST:
    def __init__(self, *_):
        self._X = np.arange(10 * 784).reshape(10, 784).astype(np.uint8)
        self._y = np.arange(10)

    def load_training(self):
        return self._X, self._y

    def load_testing(self):
        return self._X, self._y

@pytest.fixture
def tiny_mnist(monkeypatch):
    """Replace MNIST loader with a 10‑sample fake dataset (digits 0‑9 once)."""
    import src.knn.knn as knn_mod
    monkeypatch.setattr(
        knn_mod,"MNIST", DummyMNIST, raising=True
    )  # ensures we’re patching the right symbol


@pytest.fixture
def dummy_knn(monkeypatch):
    """Stub sklearn’s KNeighborsClassifier so we skip real training."""
    import src.knn.knn as knn_mod
    monkeypatch.setattr(knn_mod,"KNeighborsClassifier", FakeKNN, raising=True)


@pytest.fixture(autouse=True)
def mute_matplotlib(monkeypatch):
    """Prevent plt.show() from opening a window during test run."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *_, **__: None)
