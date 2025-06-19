# 🧠 Handwritten Digit Recognition using Machine Learning and Deep Learning

This project applies various supervised learning algorithms (KNN, SVM, Random Forest) and deep learning (CNN using Keras/TensorFlow) to recognize handwritten digits from the MNIST dataset. This was done as an attempt to integrate Software Maintenance practices into a Machine Learning project as a coursework for SWE 4802: Software Maintenance at Islamic University of Technology. The course was conducted by [Asst. Professor Lutfun Nahar Lota](https://cse.iutoic-dhaka.edu/profile/lota/education) and [Md. Rafid Haque, Lecturer](https://cse.iutoic-dhaka.edu/profile/rafidhaque/education)

---

## 📚 Table of Contents

* [🔧 Requirements](#-requirements)
* [🚀 Setup Instructions](#-setup-instructions)
* [🧪 Running Models](#-running-models)
* [📈 Code Coverage](#-code-coverage)

  * [Usage](#usage)
  * [Examples](#examples)
  * [Output](#output)
  * [Troubleshooting](#troubleshooting)
* [📂 Project Structure](#-project-structure)
* [📎 References](#-references)

---

## 🔧 Requirements

This project has been tested with:

* Python 3.9
* Conda
* scikit-learn
* numpy (with MKL on Windows)
* matplotlib
* keras
* tensorflow
* opencv-python
* pytest
* pytest-cov
* sonar-scanner
* scalene
* codecov

To install dependencies automatically, follow the setup instructions below.

---

## 🚀 Setup Instructions

1. 📥 Clone the repository:

```bash
git clone <your-repo-url>
cd Handwritten-Digit-Recognition-using-Deep-Learning
```

2. 🐍 Create the Conda environment:

```bash
conda env create -f environment.yml
```

3. ✅ Activate the environment:

```bash
conda activate swe4802
```

4. 📦 Download the MNIST dataset files:

```bash
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

5. 📂 Unzip and place the files in:

```
loader/
└── dataset/
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

6. 🔧 Install as editable package:

```bash
pip install -e .
```

---

## 🧪 Running Models

You can run each model using:

```bash
python -m src.knn.knn
python -m src.svm.svm
python -m src.rfc.rfc
python -m src.cnn.cnn
```

---

## 📈 Code Coverage

The run\_cov.sh script provides an easy way to run model-specific unit tests and generate coverage reports using pytest and pytest-cov.

### Usage

```bash
./run_cov.sh <model> [additional pytest args]
```

Where:

* <model> is one of: svm, rfc, knn, cnn
* \[additional pytest args] are optional flags passed to pytest

### Examples

Run coverage for the CNN model:

```bash
./run_cov.sh cnn
```

Run tests for SVM with verbose output:

```bash
./run_cov.sh svm -v
```

Filter tests for KNN using a keyword:

```bash
./run_cov.sh knn -k predict
```

### Output

Each run produces:

* 📜 Terminal coverage summary
* 🌐 HTML report: coverage/<model>/index.html
* 📄 XML report: coverage/<model>/coverage.xml

Example:

coverage/
├── cnn/
│   ├── index.html        ← Open in browser
│   └── coverage.xml      ← For CI tools
├── knn/
├── rfc/
└── svm/

### Troubleshooting

* ❗ Make sure all modules inside src/ have **init**.py.
* ❗ If ImportError occurs, ensure PYTHONPATH includes src/
* ❗ If you renamed model folders or scripts, update the script accordingly.

Install dependencies for testing:

```bash
pip install pytest pytest-cov
```

---

## 📂 Project Structure (Simplified)

src/
├── knn/
├── svm/
├── rfc/
├── cnn/
│   ├── cnn\_classifier.py
│   ├── neural\_network.py
│   └── tests/
└── loader/
└── dataset/

---

## 📎 References

* MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* Original Deep Learning implementation: [anujdutt9 GitHub Repo](https://github.com/anujdutt9/Handwritten-Digit-Recognition-using-Deep-Learning)