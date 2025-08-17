# Text Recognition using Machine Learning (EMNIST + SVM)

A project that trains a handwritten text recognizer on the EMNIST dataset using classical machine learning (SVM). Includes data pipeline, training, evaluation, inference scripts, and a clear README explaining technical details, outcomes, and impact.

---

## Project Overview

This repository demonstrates a complete pipeline for handwritten character recognition using the EMNIST dataset and Support Vector Machines (SVMs). Unlike deep learning examples, this project focuses on a compact, interpretable classical ML approach that is lightweight and simple to reproduce on a CPU.

Key features:
- Data loading and preprocessing (EMNIST `balanced` split)
- Dimensionality reduction (PCA) to speed up SVM training
- SVM training with hyperparameter options (LinearSVC or SVC with RBF)
- Evaluation scripts (accuracy, classification report, confusion matrix)
- Inference script to predict characters from an input image
- Full instructions for reproduction and tips for improving performance

---

## Repository structure

```
text-recognition-svm-emnist/
├── README.md                 # This file
├── requirements.txt
├── data/                     # downloads (gitignored)
├── models/
│   └── svm_emnist.joblib     # saved model (created after training)
├── src/
│   ├── data_prep.py         # dataset loading & preprocessing utilities
│   ├── train_svm.py         # training script
│   ├── evaluate.py          # evaluation and metrics
│   └── predict.py           # inference on a single image
├── notebooks/
│   └── EDA_and_training.ipynb
├── .gitignore
├── LICENSE
└── CONTRIBUTING.md
```

---

## Requirements

Install with pip (use a virtualenv):

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
numpy
scikit-learn
matplotlib
pillow
joblib
torch
torchvision
tqdm
pandas
seaborn
```

> Note: `torch` and `torchvision` are only used to download and load the EMNIST dataset conveniently; the model itself uses scikit-learn.

---

## Data

We use the EMNIST `balanced` split which contains a mixture of digits and letters. The `torchvision.datasets.EMNIST` class will download the data automatically the first time you run the scripts.

**Important preprocessing notes**:
- EMNIST images are 28x28 and **rotated** relative to visual expectations; the pipeline in `data_prep.py` corrects orientation and normalizes pixel values.
- Images are flattened for the SVM (SVM accepts 1D feature vectors). PCA is applied to reduce dimensionality and improve CPU training time.

---

## How to run

1. Clone repo

```bash
git clone https://github.com/Amruth-Reddy-N/text-recognition-svm-emnist.git
cd text-recognition-svm-emnist
```

2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Train model (default PCA=150, RBF SVM)

```bash
python src/train_svm.py --eval
```

This will download EMNIST (if missing), train the pipeline, save the model to `models/svm_emnist.joblib`, and print test accuracy.

To run a grid search (slower):

```bash
python src/train_svm.py --grid_search --eval
```

4. Evaluate with plots

```bash
python src/evaluate.py --model ./models/svm_emnist.joblib
```

5. Predict on a single image

```bash
python src/predict.py ./examples/sample_A.png
```

---

## Expected outcomes & sample results

- A lightweight SVM-based recognizer that can reach **~85–92%** test accuracy on EMNIST `balanced` depending on PCA components and hyperparameters. (Exact numbers vary by machine and hyperparameter choices.)
- Model file `models/svm_emnist.joblib` that can be loaded quickly for inference on new images.
- Confusion matrix and classification report helping identify frequently confused character pairs (e.g., `O` vs `0`, `I` vs `l`).

> These numbers are *expected ranges* for a classical pipeline; deep learning (CNN) approaches will typically achieve higher accuracy but require GPUs and more code.

---

## Impact and discussion

- **Accessibility:** This project shows how to build a capable OCR system without heavy deep learning infrastructure. This lowers the barrier for educational and low-resource applications.
- **Interpretability:** SVMs + PCA are more interpretable than deep networks and are useful in environments where explainability matters.
- **Deployability:** The saved scikit-learn pipeline is small and CPU-friendly, suitable for integration into desktop apps or lightweight servers.

Limitations:
- Performance is usually lower than modern CNN-based solutions.
- SVM training (with non-linear kernels) scales poorly to very large datasets — PCA helps mitigate this.

---
