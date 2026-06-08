"""
train.py
--------
Unified training entry point. Train any model (or all models) with a
single command-line argument.

Usage:
    python src/train.py --model lr      # Logistic Regression
    python src/train.py --model svm     # Support Vector Machine
    python src/train.py --model lstm    # Bidirectional LSTM
    python src/train.py --model muril   # MuRIL transformer
    python src/train.py --model all     # All four models in sequence
"""

import os
import sys
import argparse
import random

import numpy as np

# ---------------------------------------------------------------------------
# Fix all random seeds before any imports that use randomness
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass  # torch not required for classical models

# ---------------------------------------------------------------------------
# Add src/ to path so relative imports work when running from project root
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import load_clean_data, split_data


# ---------------------------------------------------------------------------
# Per-model training routines
# ---------------------------------------------------------------------------

def run_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate Logistic Regression on TF-IDF features."""
    from features import get_tfidf_features
    from models.classical import train_logistic_regression, evaluate_model

    print("\n" + "=" * 60)
    print("  MODEL: Logistic Regression")
    print("=" * 60)
    train_tfidf, val_tfidf, test_tfidf = get_tfidf_features(X_train, X_val, X_test)
    model = train_logistic_regression(train_tfidf, y_train)
    evaluate_model(model, test_tfidf, y_test, model_name="Logistic Regression")


def run_svm(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate LinearSVC on TF-IDF features."""
    from features import get_tfidf_features
    from models.classical import train_svm, evaluate_model

    print("\n" + "=" * 60)
    print("  MODEL: SVM (LinearSVC)")
    print("=" * 60)
    train_tfidf, val_tfidf, test_tfidf = get_tfidf_features(X_train, X_val, X_test)
    model = train_svm(train_tfidf, y_train)
    evaluate_model(model, test_tfidf, y_test, model_name="SVM")


def run_lstm(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate Bidirectional LSTM."""
    import tensorflow as tf
    tf.random.set_seed(SEED)

    from features import get_lstm_sequences
    from models.lstm_model import (
        build_lstm_model, train_lstm, plot_training_history, evaluate_lstm,
    )

    print("\n" + "=" * 60)
    print("  MODEL: Bidirectional LSTM")
    print("=" * 60)

    # Hyperparameters — document these in your paper
    MAX_WORDS = 30_000
    MAX_LEN   = 100

    train_seq, val_seq, test_seq = get_lstm_sequences(
        X_train, X_val, X_test,
        max_words=MAX_WORDS,
        max_len=MAX_LEN,
    )

    model = build_lstm_model(
        vocab_size=MAX_WORDS + 1,  # +1 for the OOV token
        embedding_dim=128,
        lstm_units=128,
        num_classes=3,
    )

    model, history = train_lstm(model, train_seq, y_train, val_seq, y_val)
    plot_training_history(history)
    evaluate_lstm(model, test_seq, y_test)


def run_muril(X_train, X_val, X_test, y_train, y_val, y_test):
    """Fine-tune and evaluate MuRIL."""
    from features import get_transformer_tokenizer
    from models.transformer_model import (
        prepare_hf_dataset, load_muril_model, train_muril, evaluate_muril,
    )

    print("\n" + "=" * 60)
    print("  MODEL: MuRIL (google/muril-base-cased)")
    print("=" * 60)
    print("  NOTE: GPU recommended. On CPU this may take several hours.")
    print("  Tip:  Use Google Colab (Runtime > Change runtime type > GPU)")
    print("=" * 60)

    import torch

    tokenizer = get_transformer_tokenizer()
    max_length = 64 if not torch.cuda.is_available() else 128
    train_ds, val_ds, test_ds = prepare_hf_dataset(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    model   = load_muril_model(num_labels=3)
    trainer = train_muril(model, train_ds, val_ds)
    evaluate_muril(trainer, test_ds, y_test)


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hinglish Sentiment Analysis — Model Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/train.py --model lr      # Logistic Regression baseline
  python src/train.py --model svm     # SVM baseline
  python src/train.py --model lstm    # Bidirectional LSTM
  python src/train.py --model muril   # MuRIL transformer (GPU recommended)
  python src/train.py --model all     # Train all models sequentially
        """,
    )
    parser.add_argument(
        "--model",
        choices=["lr", "svm", "lstm", "muril", "all"],
        required=True,
        help="Which model to train.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Step 1 — Load data and create the shared split
    # All models MUST use the exact same split (random_state=42).
    # -----------------------------------------------------------------------
    texts, labels = load_clean_data("data/processed/clean_data.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels)

    # -----------------------------------------------------------------------
    # Step 2 — Dispatch to the chosen model
    # -----------------------------------------------------------------------
    dispatch = {
        "lr":    run_logistic_regression,
        "svm":   run_svm,
        "lstm":  run_lstm,
        "muril": run_muril,
    }

    if args.model == "all":
        print("\n[train.py] Training all models in sequence ...\n")
        for name, fn in dispatch.items():
            fn(X_train, X_val, X_test, y_train, y_val, y_test)
    else:
        dispatch[args.model](X_train, X_val, X_test, y_train, y_val, y_test)

    print("\n[train.py] Done. Check results/ for metrics and plots.")


if __name__ == "__main__":
    main()
