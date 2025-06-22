import sys
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pytest
from loader.data_loader import DummyMNIST
from src.knn.tests.fakes import FakeKNN




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



@pytest.fixture
def dummy_knn(monkeypatch):
    """
    Swap out scikit-learn's KNeighborsClassifier for our ultra-fast FakeKNN,
    so tests run instantly and headless CI isn’t slowed down.
    """
    import src.knn.knn as knn_mod
    monkeypatch.setattr(knn_mod, "KNeighborsClassifier", FakeKNN, raising=True)


@pytest.fixture(autouse=True)
def mute_matplotlib(monkeypatch):
    """Prevent GUI pop-ups (plt.show) during the test run."""
    monkeypatch.setattr(plt, "show", lambda *_, **__: None)
