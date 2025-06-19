from pathlib import Path
from typing import Tuple, Union
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection,svm
from sklearn.metrics import accuracy_score, confusion_matrix
from loader.mnist_loader import MNIST
from matplotlib import style

style.use("ggplot")


def load_data(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = MNIST(dataset_dir)
    train_img, train_labels = map(np.array, data.load_training())
    test_img,  test_labels  = map(np.array, data.load_testing())
    return train_img, train_labels, test_img, test_labels


def train_and_pickle(
    x_train: np.ndarray,
    y_train: np.ndarray,
    pickle_path: Union[str, Path],
    gamma: float = 0.1,
    kernel: str = "poly",
) -> Path:
    clf = svm.SVC(gamma=gamma, kernel=kernel)
    clf.fit(x_train, y_train)

    p = Path(pickle_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(clf, f)
    return p


def evaluate(clf, x: np.ndarray, y: np.ndarray):
    preds = clf.predict(x)
    return accuracy_score(y, preds), confusion_matrix(y, preds), preds


def plot_cm(cm: np.ndarray, title: str):
    plt.matshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.axis("off")
    plt.show()



def main(
    dataset_dir: str = "loader/dataset/",
    pickle_path: str = "dump/MNIST_SVM.pickle",
):
    
    train_img, train_labels, test_img, test_labels = load_data(dataset_dir)

    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        train_img, train_labels, test_size=0.1, random_state=42
    )

    model_file = train_and_pickle(x_train, y_train, pickle_path)
    with model_file.open("rb") as f:
        clf = pickle.load(f)

    val_acc, cm_val, _ = evaluate(clf, x_val, y_val)
    print("\nValidation accuracy:", val_acc)
    plot_cm(cm_val, "Confusion Matrix (Validation)")

    test_acc, cm_test, _ = evaluate(clf, test_img, test_labels)
    print("\nTest accuracy:", test_acc)
    plot_cm(cm_test, "Confusion Matrix (Test)")


if __name__ == "__main__":
    main()
