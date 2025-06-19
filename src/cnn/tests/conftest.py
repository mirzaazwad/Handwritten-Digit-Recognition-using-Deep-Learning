import numpy as np
import pytest

@pytest.fixture
def tiny_mnist():
    """
    Returns a small dummy MNIST dataset (10 images, one for each digit).
    Image size: 28x28
    """
    X = np.arange(10 * 28 * 28).reshape(10, 28, 28, 1).astype(np.float32)
    X /= 255.0
    y = np.arange(10)
    return X, y
