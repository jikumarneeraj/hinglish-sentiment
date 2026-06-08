"""
lstm_model.py
-------------
Builds and trains a Bidirectional LSTM neural network for Hinglish
sentiment classification (3 classes: positive, negative, neutral).

Architecture:
    Embedding (vocab_size × 128)
    → Bidirectional LSTM (128 units)
    → Dropout (0.4)
    → Dense (64, ReLU)
    → Dropout (0.3)
    → Dense (3, Softmax)

Run via train.py:
    python src/train.py --model lstm
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# Fix seeds for reproducibility
import random
import tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES    = ["Positive", "Negative", "Neutral"]
METRICS_PATH   = os.path.join(RESULTS_DIR, "metrics.json")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "lstm_best_model.h5")


def _load_metrics() -> dict:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_metrics(metrics: dict) -> None:
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    num_classes: int = 3,
):
    """
    Define and compile the Bidirectional LSTM architecture.

    Hyperparameters (document these in your paper's Experiments section):
        embedding_dim = 128
        lstm_units    = 128
        dropout_1     = 0.4
        dropout_2     = 0.3
        dense_units   = 64
        optimizer     = Adam
        loss          = sparse_categorical_crossentropy

    Parameters
    ----------
    vocab_size    : int   size of the vocabulary (from Keras Tokenizer)
    embedding_dim : int   dimensionality of the embedding layer
    lstm_units    : int   number of LSTM units per direction
    num_classes   : int   number of output classes (3)

    Returns
    -------
    Compiled Keras Sequential model
    """
    model = Sequential([
        # Layer 1 — Embedding: learns a dense vector for each token
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding",
        ),
        # Layer 2 — Bidirectional LSTM: reads text forward AND backward
        Bidirectional(
            LSTM(units=lstm_units, return_sequences=False),
            name="bidirectional_lstm",
        ),
        # Layer 3 — Dropout: prevents overfitting
        Dropout(0.4, name="dropout_1"),
        # Layer 4 — Dense: learns combinations of LSTM features
        Dense(64, activation="relu", name="dense_hidden"),
        # Layer 5 — Dropout
        Dropout(0.3, name="dropout_2"),
        # Layer 6 — Output: 3-class softmax
        Dense(num_classes, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lstm(model, X_train, y_train, X_val, y_val):
    """
    Train the LSTM with early stopping and model checkpointing.

    Hyperparameters:
        batch_size = 32
        epochs     = 20  (early stopping typically halts earlier)
        patience   = 3   (stop if val_loss does not improve for 3 epochs)

    Parameters
    ----------
    model   : compiled Keras model from build_lstm_model()
    X_train : np.ndarray  (n_train, max_len)
    y_train : array-like  integer labels
    X_val   : np.ndarray  (n_val, max_len)
    y_val   : array-like  integer labels

    Returns
    -------
    (trained model, history object)
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        np.array(X_train),
        np.array(y_train),
        validation_data=(np.array(X_val), np.array(y_val)),
        batch_size=32,   # hyperparameter — document in paper
        epochs=20,       # hyperparameter — early stopping controls actual epochs
        callbacks=callbacks,
        verbose=1,
    )

    # Persist training history (accuracy/loss per epoch) for the paper's appendix
    history_path = os.path.join(RESULTS_DIR, "lstm_history.json")
    history_data = {
        "accuracy":     history.history.get("accuracy", []),
        "val_accuracy": history.history.get("val_accuracy", []),
        "loss":         history.history.get("loss", []),
        "val_loss":     history.history.get("val_loss", []),
    }
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"[train_lstm] History saved to '{history_path}'.")

    return model, history


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_history(history) -> None:
    """
    Plot training vs validation accuracy and loss curves.

    Saves to results/lstm_training_curves.png.
    Use this figure in the paper's appendix or experiments section.
    """
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy subplot
    axes[0].plot(epochs, hist["accuracy"], label="Train Accuracy", marker="o")
    axes[0].plot(epochs, hist["val_accuracy"], label="Val Accuracy", marker="o")
    axes[0].set_title("LSTM — Accuracy per Epoch", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Loss subplot
    axes[1].plot(epochs, hist["loss"], label="Train Loss", marker="o")
    axes[1].plot(epochs, hist["val_loss"], label="Val Loss", marker="o")
    axes[1].set_title("LSTM — Loss per Epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "lstm_training_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot_training_history] Saved to '{save_path}'.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_lstm(model, X_test, y_test) -> dict:
    """
    Evaluate the trained LSTM on the test set.

    Appends metrics to results/metrics.json under key 'LSTM'.
    Saves confusion matrix to results/confusion_matrix_LSTM.png.

    Parameters
    ----------
    model  : trained Keras model
    X_test : np.ndarray  (n_test, max_len)
    y_test : array-like  true integer labels

    Returns
    -------
    dict  { accuracy, f1_macro, precision_macro, recall_macro }
    """
    # model.predict returns probability arrays; take argmax for class index
    y_prob = model.predict(np.array(X_test), verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.array(y_test)

    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*50}")
    print("  LSTM — Test Set Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Append to shared metrics file
    all_metrics = _load_metrics()
    all_metrics["LSTM"] = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
    }
    _save_metrics(all_metrics)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, model_name="LSTM")

    return all_metrics["LSTM"]


def _plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """Save a confusion matrix heatmap PNG."""
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

    save_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[_plot_confusion_matrix] Saved to '{save_path}'.")
