import pickle
import src.knn.knn as knn
import numpy as np
from pathlib import Path


def test_load_data(tiny_mnist):
    train_X, train_y, test_X, test_y = knn.load_data("dummy")
    assert train_X.shape == (10, 784)
    assert np.array_equal(train_y, np.arange(10))
    assert np.array_equal(test_X, train_X)
    assert np.array_equal(test_y, train_y)


def test_train_and_pickle(dummy_knn, tmp_path):
    X = np.random.randint(0, 255, size=(4, 784))
    y = np.array([0, 1, 2, 3])
    pkl = tmp_path / "knn.pkl"

    knn.train_and_pickle(X, y, pkl, n_neighbors=1, n_jobs=1)

    assert pkl.exists(), "pickle file should be created"
    clf = pickle.load(open(pkl, "rb"))
    preds = clf.predict(X)
    assert preds.shape == (4,)


def test_evaluate(tiny_mnist, dummy_knn):
    X, y, _, _ = knn.load_data("dummy")
    model = knn.KNeighborsClassifier()
    model.fit(X, y)

    acc, cm, preds = knn.evaluate(model, X, y)
    assert 0.0 <= acc <= 1.0
    assert cm.shape == (10, 10)
    assert preds.shape == (10,)


def test_end_to_end(tmp_path, tiny_mnist, dummy_knn, capsys):
    """Run knn.main() end‑to‑end with fakes – hits ~100 % of lines."""
    pkl = tmp_path / "fake.pkl"
    knn.main(dataset_dir="dummy", pickle_path=str(pkl))

    captured = capsys.readouterr().out
    assert "Validation accuracy:" in captured
    assert "Test accuracy:" in captured
