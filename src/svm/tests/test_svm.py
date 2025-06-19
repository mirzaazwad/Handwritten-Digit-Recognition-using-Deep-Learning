import numpy as np
from pathlib import Path
import src.svm.svm as svm_mod


def test_load_data(tiny_mnist):
    train_X, train_y, test_X, test_y = svm_mod.load_data("dummy")
    assert train_X.shape == (10, 784)
    assert np.array_equal(train_y, np.arange(10))
    assert np.array_equal(test_X, train_X)
    assert np.array_equal(test_y, train_y)


def test_train_and_pickle(tmp_path):
    X = np.random.randint(0, 255, size=(4, 784))
    y = np.array([0, 1, 2, 3])
    pkl = tmp_path / "svm.pkl"

    model_file = svm_mod.train_and_pickle(X, y, pkl, gamma=0.1, kernel="poly")
    assert model_file.exists()



def test_evaluate(tiny_mnist):
    X, y, _, _ = svm_mod.load_data("dummy")
    model = svm_mod.svm.SVC(gamma=0.1, kernel="poly")
    model.fit(X, y)

    acc, cm, preds = svm_mod.evaluate(model, X, y)
    assert 0.0 <= acc <= 1.0
    assert cm.shape == (10, 10)
    assert preds.shape == (10,)



def test_end_to_end(tmp_path, tiny_mnist, capsys):
    pkl = tmp_path / "fake.pkl"
    svm_mod.main(dataset_dir="dummy", pickle_path=str(pkl))

    out = capsys.readouterr().out
    assert "Validation accuracy:" in out
    assert "Test accuracy:" in out
    assert pkl.exists()
