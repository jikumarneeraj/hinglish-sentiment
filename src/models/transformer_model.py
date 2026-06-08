"""
transformer_model.py
--------------------
Fine-tunes MuRIL (google/muril-base-cased) — Google's multilingual BERT
pretrained specifically on 17 Indian languages including Hinglish.

This is the main contribution of the paper. MuRIL's pretraining on
transliterated Indian-language text gives it a significant advantage
over general-purpose multilingual models on Hinglish sentiment data.

Hardware recommendation: Use Google Colab with a T4 GPU.
    Runtime → Change runtime type → GPU
    Expected training time: 20–30 minutes (5 epochs).

Run via train.py:
    python src/train.py --model muril
"""

import os
import json

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# Fix seeds
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

RESULTS_DIR     = "results"
MURIL_CKPT_DIR  = os.path.join(RESULTS_DIR, "muril_checkpoints")
METRICS_PATH    = os.path.join(RESULTS_DIR, "metrics.json")
MODEL_NAME      = "google/muril-base-cased"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MURIL_CKPT_DIR, exist_ok=True)

CLASS_NAMES = ["Positive", "Negative", "Neutral"]


def _load_metrics() -> dict:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_metrics(metrics: dict) -> None:
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def resolve_muril_checkpoint(ckpt_dir: str = MURIL_CKPT_DIR) -> str:
    """
    Return a directory HuggingFace can load (checkpoint-* subfolder if needed).

    Trainer saves under checkpoint-*/ not the parent muril_checkpoints/ folder.
    """
    if os.path.isfile(os.path.join(ckpt_dir, "config.json")):
        return ckpt_dir

    from pathlib import Path

    checkpoints = sorted(
        Path(ckpt_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No MuRIL checkpoint found in '{ckpt_dir}'")

    best_path, best_f1 = None, -1.0
    for ckpt in checkpoints:
        state_file = ckpt / "trainer_state.json"
        if not state_file.exists():
            continue
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
        for entry in reversed(state.get("log_history", [])):
            f1 = entry.get("eval_f1")
            if f1 is not None and f1 > best_f1:
                best_f1 = f1
                best_path = ckpt
            break

    return str(best_path or checkpoints[-1])


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_hf_dataset(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    tokenizer,
    max_length: int = 128,
):
    """
    Convert text lists into HuggingFace Dataset objects.

    Tokenisation is applied via dataset.map() with:
        max_length  = 128   (covers >95% of tweets)
        truncation  = True
        padding     = 'max_length'

    Parameters
    ----------
    X_* : list[str]   raw (cleaned) text
    y_* : list[int]   integer labels
    tokenizer         HuggingFace tokenizer from get_transformer_tokenizer()
    max_length : int  token sequence length

    Returns
    -------
    train_dataset, val_dataset, test_dataset : HuggingFace Dataset
    """
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def _make_dataset(texts, labels):
        ds = Dataset.from_dict({"text": texts, "label": labels})
        ds = ds.map(_tokenize, batched=True, remove_columns=["text"])
        return ds

    train_dataset = _make_dataset(X_train, y_train)
    val_dataset   = _make_dataset(X_val, y_val)
    test_dataset  = _make_dataset(X_test, y_test)

    print(
        f"[prepare_hf_dataset] "
        f"Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_muril_model(num_labels: int = 3):
    """
    Download and load the pretrained MuRIL model with a
    3-class sequence-classification head.

    MuRIL (~900 MB) downloads automatically from HuggingFace Hub
    on the first call. Ensure internet access and sufficient disk space.

    Parameters
    ----------
    num_labels : int   number of output classes (3)

    Returns
    -------
    AutoModelForSequenceClassification
    """
    print(f"[load_muril_model] Loading '{MODEL_NAME}' ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )
    print("[load_muril_model] Model loaded.")
    return model


# ---------------------------------------------------------------------------
# Metrics callback (used by HuggingFace Trainer)
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Callback function passed to the HuggingFace Trainer.
    Called automatically at each evaluation step.

    Parameters
    ----------
    eval_pred : EvalPrediction
        Named tuple with .predictions (logits) and .label_ids.

    Returns
    -------
    dict  { accuracy, f1, precision, recall }
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy":  round(accuracy_score(labels, preds), 4),
        "f1":        round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "precision": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, average="macro", zero_division=0), 4),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_muril(model, train_dataset, val_dataset):
    """
    Fine-tune MuRIL using the HuggingFace Trainer API.

    Hyperparameters (document all of these in your paper's Experiments section):
        num_train_epochs              = 5
        per_device_train_batch_size   = 16
        per_device_eval_batch_size    = 32
        learning_rate                 = 2e-5
        warmup_steps                  = 100
        weight_decay                  = 0.01
        early_stopping_patience       = 2

    Parameters
    ----------
    model         : loaded MuRIL model from load_muril_model()
    train_dataset : HuggingFace Dataset (training split)
    val_dataset   : HuggingFace Dataset (validation split)

    Returns
    -------
    Trainer  (contains the best checkpoint model)
    """
    # CPU training is much slower; use larger batches and fewer epochs locally.
    # Full hyperparameters (5 epochs, batch 16) are used when a GPU is available.
    on_cpu = not torch.cuda.is_available()
    num_epochs = 2 if on_cpu else 5
    train_batch = 32 if on_cpu else 16
    eval_batch = 64 if on_cpu else 32

    if on_cpu:
        print(
            "[train_muril] CPU detected — using 2 epochs, batch_size=32 "
            "(use Google Colab GPU for full 5-epoch settings)."
        )

    training_args = TrainingArguments(
        output_dir=MURIL_CKPT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        learning_rate=2e-5,                      # hyperparameter — standard for BERT fine-tuning
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_safetensors=False,  # avoids non-contiguous tensor error on some Colab builds
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(RESULTS_DIR, "muril_logs"),
        logging_steps=50,
        report_to="none",                        # disable wandb/tensorboard
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("[train_muril] Starting fine-tuning ...")
    trainer.train()
    print("[train_muril] Fine-tuning complete. Best checkpoint saved.")
    return trainer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_muril(trainer, test_dataset, y_test) -> dict:
    """
    Evaluate the fine-tuned MuRIL on the held-out test set.

    Appends metrics to results/metrics.json under key 'MuRIL'.
    Saves confusion matrix to results/confusion_matrix_MuRIL.png.

    Parameters
    ----------
    trainer      : Trainer object returned by train_muril()
    test_dataset : HuggingFace Dataset (test split)
    y_test       : list[int]  true labels (needed for confusion matrix)

    Returns
    -------
    dict  { accuracy, f1_macro, precision_macro, recall_macro }
    """
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = np.array(y_test)

    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*50}")
    print("  MuRIL — Test Set Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    all_metrics = _load_metrics()
    all_metrics["MuRIL"] = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
    }
    _save_metrics(all_metrics)

    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, model_name="MuRIL")

    return all_metrics["MuRIL"]


def _plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
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
