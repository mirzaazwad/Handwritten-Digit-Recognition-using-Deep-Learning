import pickle
from pathlib import Path

import numpy as np

import src.svm.svm as svm_mod


def test_load_data(tiny_mnist):
    loader = svm_mod.DataLoader(dataset_dir="dummy")
    train_X, train_y, test_X, test_y = loader.load()

    assert train_X.shape == (10, 784)
    assert np.array_equal(train_y, np.arange(10))
    assert np.array_equal(test_X, train_X)
    assert np.array_equal(test_y, train_y)


def test_train_and_pickle(dummy_svm, tmp_path):
    X = np.random.randint(0, 255, size=(4, 784))
    y = np.array([0, 1, 2, 3])
    pkl: Path = tmp_path / "svm.pkl"

    trainer = svm_mod.SVMTrainer(gamma=0.1, kernel="poly")
    trainer.fit_and_save(X, y, pkl)

    assert pkl.exists()
    clf = pickle.load(open(pkl, "rb"))
    preds = clf.predict(X)
    assert preds.shape == (4,)


def test_evaluate(tiny_mnist, dummy_svm):
    loader = svm_mod.DataLoader(dataset_dir="dummy")
    X, y, _, _ = loader.load()

    model = svm_mod.SVC(gamma=0.1, kernel="poly")  # patched → FakeSVM
    model.fit(X, y)

    acc, cm, preds = svm_mod.Evaluator.evaluate(model, X, y)
    assert 0.0 <= acc <= 1.0
    assert cm.shape == (10, 10)
    assert preds.shape == (10,)


def test_end_to_end(tmp_path, tiny_mnist, dummy_svm, capsys):
    pkl = tmp_path / "fake.pkl"
    svm_mod.main(dataset_dir="dummy", pickle_path=str(pkl))

    out = capsys.readouterr().out
    assert "Validation accuracy:" in out
    assert "Test accuracy:" in out
    assert pkl.exists()
