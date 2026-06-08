"""
error_analysis.py
-----------------
Extract misclassified test examples for the paper's Error Analysis section.
Uses the fine-tuned MuRIL checkpoint.

Run AFTER MuRIL training:
    python src/error_analysis.py

Output:
    results/error_analysis.csv   — all misclassified test examples
    results/error_analysis.txt   — 15 random samples for the paper
"""

import os
import sys
import random

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import load_clean_data, split_data, get_transformer_tokenizer
from models.transformer_model import prepare_hf_dataset, resolve_muril_checkpoint

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = "results"
LABEL_NAMES = {0: "Positive", 1: "Negative", 2: "Neutral"}
N_SAMPLES = 15


def main():
    try:
        ckpt_path = resolve_muril_checkpoint()
    except FileNotFoundError as exc:
        print(f"[error_analysis] {exc}")
        print("  Run: python src/train.py --model muril")
        return

    print(f"[error_analysis] Loading checkpoint: {ckpt_path}")

    texts, labels = load_clean_data()
    _, _, X_test, _, _, y_test = split_data(texts, labels)

    tokenizer = get_transformer_tokenizer()
    _, _, test_ds = prepare_hf_dataset(
        [X_test[0]], [y_test[0]],
        [X_test[0]], [y_test[0]],
        X_test, y_test,
        tokenizer=tokenizer,
    )

    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=RESULTS_DIR, report_to="none"),
    )
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = np.array(y_test)

    errors = [
        {
            "text": text,
            "true_label": LABEL_NAMES[true_l],
            "predicted_label": LABEL_NAMES[pred_l],
        }
        for text, true_l, pred_l in zip(X_test, y_true, y_pred)
        if true_l != pred_l
    ]

    if not errors:
        print("[error_analysis] No misclassifications on test set.")
        return

    df = pd.DataFrame(errors)
    sample = df.sample(n=min(N_SAMPLES, len(df)), random_state=SEED)

    csv_path = os.path.join(RESULTS_DIR, "error_analysis.csv")
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(RESULTS_DIR, "error_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            f"Error Analysis — MuRIL "
            f"({len(errors)} misclassified / {len(y_test)} test)\n"
        )
        f.write("=" * 70 + "\n\n")
        for i, row in enumerate(sample.itertuples(), 1):
            f.write(f"Example {i}\n")
            f.write(f"  Text     : {row.text}\n")
            f.write(f"  True     : {row.true_label}\n")
            f.write(f"  Predicted: {row.predicted_label}\n\n")

    print(f"[error_analysis] {len(errors)} misclassified → '{csv_path}'")
    print(f"[error_analysis] {len(sample)} samples → '{txt_path}'")


if __name__ == "__main__":
    main()
