import numpy as np
from pathlib import Path
import src.rfc.rfc as rfc


def test_load_data(tiny_mnist):
    train_X, train_y, test_X, test_y = rfc.load_data("dummy")
    assert train_X.shape == (10, 784)
    assert np.array_equal(train_y, np.arange(10))
    assert np.array_equal(test_X, train_X)
    assert np.array_equal(test_y, train_y)

def test_train_and_pickle(tmp_path):
    X = np.random.randint(0, 255, size=(4, 784))
    y = np.array([0, 1, 2, 3])
    pkl = tmp_path / "rfc.pkl"
    model_file = rfc.train_and_pickle(X, y, pkl, n_estimators=5, n_jobs=1)
    assert model_file.exists()

def test_evaluate(tiny_mnist):
    X, y, _, _ = rfc.load_data("dummy")
    model = rfc.RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    acc, cm, preds = rfc.evaluate(model, X, y)
    assert 0.0 <= acc <= 1.0
    assert cm.shape == (10, 10)
    assert preds.shape == (10,)

def test_end_to_end(tmp_path, tiny_mnist, capsys):
    pkl = tmp_path / "fake.pkl"
    rfc.main(dataset_dir="dummy", pickle_path=str(pkl))
    captured = capsys.readouterr().out
    assert "Validation accuracy:" in captured
    assert "Test accuracy:" in captured
