import sys
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pytest
from loader.data_loader import DummyMNIST
from src.rfc.tests.fakes import FakeRFC




@pytest.fixture
def tiny_mnist(monkeypatch):
    """
    Replace the real MNIST loader used inside DataLoader with a 10-sample fake.
    We patch the MNIST symbol inside *loader.data_loader*, because that's
    where DataLoader looks it up.
    """
    import loader.data_loader as dl_mod          
    from src.rfc.tests.conftest import DummyMNIST  
    monkeypatch.setattr(dl_mod, "MNIST", DummyMNIST, raising=True)


@pytest.fixture
def dummy_rfc(monkeypatch):
    import src.rfc.rfc as rfc_mod

    monkeypatch.setattr(rfc_mod, "RandomForestClassifier", FakeRFC, raising=True)


@pytest.fixture(autouse=True)
def mute_matplotlib(monkeypatch):
    """Disable plt.show during tests."""
    monkeypatch.setattr(plt, "show", lambda *_, **__: None)
