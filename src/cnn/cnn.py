import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from .nn import neural_network as cnn


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = fetch_openml('mnist_784')
    mnist_data = dataset.data.to_numpy().reshape((-1, 28, 28, 1))
    mnist_data = mnist_data / 255.0 
    labels = dataset.target.astype("int")
    train_img, test_img, train_labels, test_labels = train_test_split(mnist_data, labels, test_size=0.1,random_state=42)
    return train_img, train_labels,test_img ,test_labels


def build_model(save_weights_path: Optional[str] = None):
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    clf = cnn.CNN.build(width=28, height=28, depth=1, total_classes=10, save_weights_path=save_weights_path)
    clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return clf


def train_model(model, train_img, train_labels, batch_size=128, epochs=20):
    print("Training model...")
    return model.fit(train_img, train_labels, batch_size=batch_size, epochs=epochs, verbose=1)


def evaluate_model(model, test_img, test_labels):
    print("Evaluating model...")
    _, accuracy = model.evaluate(test_img, test_labels, batch_size=128, verbose=1)
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy


def save_model(model, path: str):
    print(f"Saving model weights to {path}")
    model.save_weights(path)


def predict_and_display(model, test_img, test_labels, show=False):
    rng = np.random.default_rng(seed=42)
    for idx in rng.choice(len(test_labels), size=5):
        probs = model.predict(test_img[np.newaxis, idx])
        pred = probs.argmax(axis=1)[0]
        true = np.argmax(test_labels[idx]) if test_labels.ndim > 1 else test_labels[idx]

        print(f"Predicted: {pred}, Actual: {true}")
        if show:
            image = (test_img[idx][0] * 255).astype("uint8")
            image = cv2.merge([image] * 3)
            image = cv2.resize(image, (100, 100))
            cv2.putText(image, str(pred), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow("Digit", image)
            cv2.waitKey(0)


def main(save_model_flag=-1, load_model_flag=-1, save_weights_path="cnn_model.h5"):
    train_img, train_labels,test_img ,test_labels = load_data()

    train_labels_cat = np_utils.to_categorical(train_labels, 10)
    test_labels_cat = np_utils.to_categorical(test_labels, 10)

    model = build_model(save_weights_path if load_model_flag > 0 else None)

    if load_model_flag < 0:
        train_model(model, train_img, train_labels_cat)

    evaluate_model(model, test_img, test_labels_cat)

    if save_model_flag > 0:
        save_model(model, save_weights_path)

    predict_and_display(model, test_img, test_labels_cat, show=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_model", type=int, default=-1)
    parser.add_argument("-l", "--load_model", type=int, default=-1)
    parser.add_argument("-w", "--save_weights", type=str, default="cnn_model.h5")
    args = parser.parse_args()

    main(
        save_model_flag=args.save_model,
        load_model_flag=args.load_model,
        save_weights_path=args.save_weights
    )
