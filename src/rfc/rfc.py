"""
MNIST + Random-Forest pipeline (class-based)
-------------------------------------------
• DataLoader   – loads MNIST
• RFCTrainer   – builds, fits, pickles a RandomForestClassifier
• Evaluator    – accuracy / confusion-matrix helpers
• Visualizer   – simple CM plot (matplotlib)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier as _RFC  # real class
from sklearn.metrics import accuracy_score, confusion_matrix

from loader.data_loader import DataLoader

style.use("ggplot")
RandomForestClassifier = _RFC




class RFCTrainer:
    def __init__(
        self,
        n_estimators: int = 100,
        n_jobs: int = 10,
        random_state: int = 42,
        min_samples_leaf: int = 3,
        max_features: str = "sqrt",
    ):
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model: RandomForestClassifier | None = None


    def build(self) -> RandomForestClassifier:
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
        )
        return self.model


    def fit_and_save(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        pickle_path: Union[str, Path] = "dump/MNIST_RFC.pickle",
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
    def load_from_pickle(pickle_file: Union[str, Path]) -> RandomForestClassifier:
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
    pickle_path: str = "dump/MNIST_RFC.pickle",
) -> None:
    train_x, train_y, test_x, test_y = DataLoader(dataset_dir).load()

    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        train_x, train_y, test_size=0.1, random_state=42
    )

    trainer = RFCTrainer(n_estimators=100, n_jobs=10)
    model_file = trainer.fit_and_save(x_train, y_train, pickle_path)

    clf = RFCTrainer.load_from_pickle(model_file)

    val_acc, cm_val, _ = Evaluator.evaluate(clf, x_val, y_val)
    print("\nValidation accuracy:", val_acc)
    Visualizer.plot_confusion_matrix(cm_val, "Confusion Matrix (Validation)")

    test_acc, cm_test, _ = Evaluator.evaluate(clf, test_x, test_y)
    print("\nTest accuracy:", test_acc)
    Visualizer.plot_confusion_matrix(cm_test, "Confusion Matrix (Test)")


if __name__ == "__main__":
    main()
