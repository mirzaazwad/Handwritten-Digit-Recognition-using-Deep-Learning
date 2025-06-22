import pickle
from pathlib import Path
import numpy as np
import src.knn.knn as knn


def test_load_data(tiny_mnist):
    """DataLoader should return the 10-sample dummy dataset."""
    loader = knn.DataLoader(dataset_dir="dummy")
    train_X, train_y, test_X, test_y = loader.load()

    assert train_X.shape == (10, 784)
    assert np.array_equal(train_y, np.arange(10))
    assert np.array_equal(test_X, train_X)
    assert np.array_equal(test_y, train_y)


def test_train_and_pickle(dummy_knn, tmp_path):
    """KNNTrainer.fit_and_save should create a pickle containing the model."""
    X = np.random.randint(0, 255, size=(4, 784))
    y = np.array([0, 1, 2, 3])
    pkl: Path = tmp_path / "knn.pkl"

    trainer = knn.KNNTrainer(n_neighbors=1, n_jobs=1)
    trainer.fit_and_save(X, y, pkl)

    assert pkl.exists(), "pickle file should be created"

    clf = pickle.load(open(pkl, "rb"))
    preds = clf.predict(X)
    assert preds.shape == (4,)


def test_evaluate(tiny_mnist, dummy_knn):
    """Evaluator should return sane shapes and accuracy range."""
    loader = knn.DataLoader(dataset_dir="dummy")
    X, y, _, _ = loader.load()

    model = knn.KNeighborsClassifier()  
    model.fit(X, y)

    acc, cm, preds = knn.Evaluator.evaluate(model, X, y)

    assert 0.0 <= acc <= 1.0
    assert cm.shape == (10, 10)
    assert preds.shape == (10,)


def test_end_to_end(tmp_path, tiny_mnist, dummy_knn, capsys):
    """
    Full pipeline smoke-test: run main() with fakes, capture stdout,
    and check that both validation and test metrics were printed.
    """
    pkl = tmp_path / "fake.pkl"
    knn.main(dataset_dir="dummy", pickle_path=str(pkl))

    captured = capsys.readouterr().out
    assert "Validation accuracy:" in captured
    assert "Test accuracy:" in captured
