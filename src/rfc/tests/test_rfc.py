import pickle
from pathlib import Path

import numpy as np

import src.rfc.rfc as rfc_mod


def test_load_data(tiny_mnist):
    loader = rfc_mod.DataLoader(dataset_dir="dummy")
    train_X, train_y, test_X, test_y = loader.load()

    assert train_X.shape == (10, 784)
    assert np.array_equal(train_y, np.arange(10))
    assert np.array_equal(test_X, train_X)
    assert np.array_equal(test_y, train_y)


def test_train_and_pickle(dummy_rfc, tmp_path):
    X = np.random.randint(0, 255, size=(4, 784))
    y = np.array([0, 1, 2, 3])
    pkl: Path = tmp_path / "rfc.pkl"

    trainer = rfc_mod.RFCTrainer(n_estimators=5, n_jobs=1)
    trainer.fit_and_save(X, y, pkl)

    assert pkl.exists()
    clf = pickle.load(open(pkl, "rb"))
    preds = clf.predict(X)
    assert preds.shape == (4,)


def test_evaluate(tiny_mnist, dummy_rfc):
    loader = rfc_mod.DataLoader(dataset_dir="dummy")
    X, y, _, _ = loader.load()

    model = rfc_mod.RandomForestClassifier(n_estimators=10) 
    model.fit(X, y)

    acc, cm, preds = rfc_mod.Evaluator.evaluate(model, X, y)
    assert 0.0 <= acc <= 1.0
    assert cm.shape == (10, 10)
    assert preds.shape == (10,)


def test_end_to_end(tmp_path, tiny_mnist, dummy_rfc, capsys):
    pkl = tmp_path / "fake.pkl"
    rfc_mod.main(dataset_dir="dummy", pickle_path=str(pkl))

    captured = capsys.readouterr().out
    assert "Validation accuracy:" in captured
    assert "Test accuracy:" in captured
    assert pkl.exists()
