"""
MNIST + SVM pipeline (class-based)
----------------------------------
• DataLoader  – loads raw MNIST
• SVMTrainer  – builds, fits, pickles an sklearn SVC
• Evaluator   – accuracy / confusion-matrix helpers
• Visualizer  – simple CM plot
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
from sklearn.svm import SVC as _SVC  

from loader.data_loader import DataLoader

style.use("ggplot")
SVC = _SVC



class SVMTrainer:
    def __init__(self, gamma: float = 0.1, kernel: str = "poly"):
        self.gamma = gamma
        self.kernel = kernel
        self.model: SVC | None = None

    def build(self) -> SVC:
        self.model = SVC(gamma=self.gamma, kernel=self.kernel)
        return self.model

    def fit_and_save(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        pickle_path: Union[str, Path] = "dump/MNIST_SVM.pickle",
    ) -> Path:
        if self.model is None:
            self.build()

        assert self.model is not None
        self.model.fit(x_train, y_train)

        p = Path(pickle_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self.model, f)
        return p

    @staticmethod
    def load_from_pickle(pickle_file: Union[str, Path]) -> SVC:
        with Path(pickle_file).open("rb") as f:
            return pickle.load(f)



class Evaluator:
    @staticmethod
    def evaluate(
        clf, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        preds = clf.predict(x)
        return accuracy_score(y, preds), confusion_matrix(y, preds), preds


class Visualizer:
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix"):
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
) -> None:
    train_x, train_y, test_x, test_y = DataLoader(dataset_dir).load()

    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        train_x, train_y, test_size=0.1, random_state=42
    )

    trainer = SVMTrainer(gamma=0.1, kernel="poly")
    model_file = trainer.fit_and_save(x_train, y_train, pickle_path)

    clf = SVMTrainer.load_from_pickle(model_file)

    val_acc, cm_val, _ = Evaluator.evaluate(clf, x_val, y_val)
    print("\nValidation accuracy:", val_acc)
    Visualizer.plot_confusion_matrix(cm_val, "Confusion Matrix (Validation)")

    test_acc, cm_test, _ = Evaluator.evaluate(clf, test_x, test_y)
    print("\nTest accuracy:", test_acc)
    Visualizer.plot_confusion_matrix(cm_test, "Confusion Matrix (Test)")


if __name__ == "__main__":
    main()
