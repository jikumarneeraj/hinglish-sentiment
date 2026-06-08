"""
classical.py
------------
Trains two classical ML baselines on TF-IDF features:
    1. Logistic Regression
    2. LinearSVC (SVM)

These are your comparison baselines. The transformer model must
outperform these to demonstrate its value on Hinglish text.

Run via train.py:
    python src/train.py --model lr
    python src/train.py --model svm
"""

import os
import json

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# Set random seed for reproducibility
import random
random.seed(42)
np.random.seed(42)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ["Positive", "Negative", "Neutral"]
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")


def _load_metrics() -> dict:
    """Load existing metrics.json or return empty dict."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_metrics(metrics: dict) -> None:
    """Persist metrics dict to results/metrics.json."""
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier on TF-IDF features.

    Hyperparameters:
        C           = 1.0    (regularisation strength)
        max_iter    = 1000   (increased to ensure convergence on sparse data)
        random_state= 42

    Parameters
    ----------
    X_train : scipy sparse matrix
        TF-IDF feature matrix for the training set.
    y_train : array-like
        Integer labels (0 = positive, 1 = negative, 2 = neutral).

    Returns
    -------
    LogisticRegression (fitted)
    """
    print("[train_logistic_regression] Training...")
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    save_path = os.path.join(RESULTS_DIR, "logistic_regression.pkl")
    joblib.dump(model, save_path)
    print(f"[train_logistic_regression] Model saved to '{save_path}'.")
    return model


def train_svm(X_train, y_train):
    """
    Train a LinearSVC (Support Vector Machine) classifier.

    Hyperparameters:
        C        = 1.0    (regularisation parameter)
        max_iter = 2000   (increased for TF-IDF sparse data)

    Parameters
    ----------
    X_train : scipy sparse matrix
    y_train : array-like

    Returns
    -------
    LinearSVC (fitted)
    """
    print("[train_svm] Training...")
    model = LinearSVC(max_iter=2000, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    save_path = os.path.join(RESULTS_DIR, "svm_model.pkl")
    joblib.dump(model, save_path)
    print(f"[train_svm] Model saved to '{save_path}'.")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a trained model on the test set.

    Computes: Accuracy, F1-macro, Precision-macro, Recall-macro.
    Prints a full per-class classification report.
    Appends results to results/metrics.json.
    Saves a confusion-matrix PNG.

    Parameters
    ----------
    model       : fitted sklearn estimator
    X_test      : scipy sparse matrix (TF-IDF test features)
    y_test      : array-like (true integer labels)
    model_name  : str  e.g. "Logistic Regression" or "SVM"

    Returns
    -------
    dict  { accuracy, f1_macro, precision_macro, recall_macro }
    """
    y_pred = model.predict(X_test)

    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Test Set Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Append to shared metrics file
    all_metrics = _load_metrics()
    all_metrics[model_name] = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
    }
    _save_metrics(all_metrics)
    print(f"[evaluate_model] Metrics saved to '{METRICS_PATH}'.")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, y_test, y_pred, model_name)

    return all_metrics[model_name]


def plot_confusion_matrix(cm, y_true, y_pred, model_name: str) -> None:
    """
    Generate and save a normalised confusion matrix heatmap.

    The figure is saved to results/confusion_matrix_{model_name}.png
    at 200 DPI and is ready for paper inclusion.

    Parameters
    ----------
    cm         : np.ndarray   confusion matrix from sklearn
    y_true     : array-like
    y_pred     : array-like
    model_name : str
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Sanitise model name for filename
    safe_name = model_name.replace(" ", "_")
    save_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{safe_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot_confusion_matrix] Saved to '{save_path}'.")
