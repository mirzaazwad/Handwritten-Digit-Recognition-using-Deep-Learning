import numpy as np
import pytest
from keras.utils import to_categorical
from src.cnn.cnn import cnn as cnn
from src.cnn.cnn import DataLoader, ModelBuilder, Trainer, Predictor


def test_load_data():
    train_X, train_y, test_X, test_y = DataLoader().load_data()
    assert train_X.shape[1:] == (28, 28, 1)
    assert test_X.shape[1:] == (28, 28, 1)
    assert len(train_X) == len(train_y)
    assert len(test_X) == len(test_y)
    assert train_X.max() <= 1.0
    assert train_X.min() >= 0.0


def test_build_model():
    model = ModelBuilder().build()
    assert model is not None
    assert hasattr(model, "predict")


def test_train_and_evaluate(tiny_mnist):
    X, y = tiny_mnist
    model = ModelBuilder().build()
    y_cat = to_categorical(y, 10)

    # Train on tiny dummy data
    model.fit(X, y_cat, batch_size=2, epochs=1, verbose=0)

    # Evaluate
    acc = Trainer().evaluate(model, X, y_cat)
    assert 0.0 <= acc <= 1.0


def test_predict_and_display(tiny_mnist):
    X, y = tiny_mnist
    model = ModelBuilder().build()
    y_cat = to_categorical(y, 10)
    model.fit(X, y_cat, batch_size=2, epochs=1, verbose=0)

    # Should not raise error
    Predictor().predict_and_display(model, X, y_cat, show=False)
