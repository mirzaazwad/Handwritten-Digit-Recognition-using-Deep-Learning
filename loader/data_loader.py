import numpy as np
from typing import Tuple
from .mnist_loader import MNIST
import pytest


class DataLoader:
    """Loads the (u)MNIST data stored with the classic `MNIST` helper."""

    def __init__(self, dataset_dir: str = "loader/dataset/") -> None:
        self.dataset_dir = dataset_dir

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (train_x, train_y, test_x, test_y)."""
        data = MNIST(self.dataset_dir)
        train_img, train_labels = map(np.array, data.load_training())
        test_img, test_labels = map(np.array, data.load_testing())
        return train_img, train_labels, test_img, test_labels
    

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
    """
    Replace the real MNIST loader used inside DataLoader with a 10-sample fake.
    We patch the MNIST symbol inside *loader.data_loader*, because that's
    where DataLoader looks it up.
    """
    import loader.data_loader as dl_mod          
    from src.knn.tests.conftest import DummyMNIST  
    monkeypatch.setattr(dl_mod, "MNIST", DummyMNIST, raising=True)