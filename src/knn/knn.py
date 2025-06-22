"""
MNIST + K-NN pipeline (class-based version)
------------------------------------------
• DataLoader      – loads raw MNIST files from `loader/mnist_loader.py`
• KNNTrainer       – trains & pickles a KNeighbors model
• Evaluator        – accuracy + confusion-matrix utilities
• Visualizer       – plotting helpers (currently just a CM plot)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from loader.data_loader import DataLoader

style.use("ggplot") 



class KNNTrainer:
    """Handles model creation, fitting and pickling."""

    def __init__(self, n_neighbors: int = 5, n_jobs: int = 10):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.model: KNeighborsClassifier | None = None

    def build(self) -> KNeighborsClassifier:
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, algorithm="auto", n_jobs=self.n_jobs
        )
        return self.model

    def fit_and_save(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        pickle_path: Union[str, Path] = "dump/MNIST_KNN.pickle",
    ) -> Path:
        """Fit on data and pickle the trained estimator."""
        if self.model is None:
            self.build()

        assert self.model is not None  # for type checkers
        self.model.fit(x_train, y_train)

        p = Path(pickle_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self.model, f)

        return p

    @staticmethod
    def load_from_pickle(pickle_file: Union[str, Path]) -> KNeighborsClassifier:
        with Path(pickle_file).open("rb") as f:
            return pickle.load(f)


class Evaluator:
    """Accuracy & confusion-matrix utilities (static mix-in style)."""

    @staticmethod
    def evaluate(
        clf, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        preds = clf.predict(x)
        return accuracy_score(y, preds), confusion_matrix(y, preds), preds



class Visualizer:
    """Simple plotting helpers for exploratory runs / reports."""

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> None:
        plt.matshow(cm)
        plt.title(title)
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.axis("off")
        plt.show()


def main(
    dataset_dir: str = "loader/dataset/",
    pickle_path: str = "dump/MNIST_KNN.pickle",
) -> None:
    train_x, train_y, test_x, test_y = DataLoader(dataset_dir).load()

    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        train_x, train_y, test_size=0.1, random_state=42
    )

    trainer = KNNTrainer(n_neighbors=5)
    model_file = trainer.fit_and_save(x_train, y_train, pickle_path)

    clf = KNNTrainer.load_from_pickle(model_file)

    val_acc, cm_val, _ = Evaluator.evaluate(clf, x_val, y_val)
    print("\nValidation accuracy:", val_acc)
    Visualizer.plot_confusion_matrix(cm_val, "Confusion Matrix (Validation)")

    test_acc, cm_test, _ = Evaluator.evaluate(clf, test_x, test_y)
    print("\nTest accuracy:", test_acc)
    Visualizer.plot_confusion_matrix(cm_test, "Confusion Matrix (Test)")


if __name__ == "__main__":
    main()
