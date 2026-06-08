"""
evaluate.py
-----------
Run AFTER training all four models. Reads results/metrics.json and
generates all comparison figures and tables for the paper.

Outputs:
    results/model_comparison_chart.png   ← main paper figure
    results/all_confusion_matrices.png   ← 2×2 grid (paper figure)
    results/results_table.csv            ← numbers for your paper table

Run:
    python src/evaluate.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

RESULTS_DIR  = "results"
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

# Display order and colour palette for consistency across all figures
MODEL_ORDER  = ["Logistic Regression", "SVM", "LSTM", "MuRIL"]
METRIC_ORDER = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
REQUIRED_METRICS = set(METRIC_ORDER)
METRIC_LABELS = {
    "accuracy":        "Accuracy",
    "f1_macro":        "F1 (Macro)",
    "precision_macro": "Precision (Macro)",
    "recall_macro":    "Recall (Macro)",
}
COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------

def _is_model_entry(name: str, scores) -> bool:
    """Return True only for trained-model metric dicts (skip JSON metadata keys)."""
    return (
        isinstance(scores, dict)
        and REQUIRED_METRICS.issubset(scores.keys())
    )


def load_all_metrics(filepath: str = METRICS_PATH) -> dict:
    """
    Load results/metrics.json written by the model training scripts.

    Returns
    -------
    dict  keyed by model name, each value is a dict of metric scores.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"'{filepath}' not found.\n"
            "Run all four models before calling evaluate.py:\n"
            "  python src/train.py --model all"
        )

    with open(filepath, "r") as f:
        raw = json.load(f)

    metrics = {k: v for k, v in raw.items() if _is_model_entry(k, v)}

    print("[load_all_metrics] Models evaluated so far:")
    for name in MODEL_ORDER:
        if name in metrics:
            print(f"  {name:<25} F1={metrics[name]['f1_macro']:.4f}")

    missing = [m for m in MODEL_ORDER if m not in metrics]
    if missing:
        print(f"\n  WARNING: Results not found for: {missing}")
        print("  Run those models before generating the full comparison.")

    return metrics


# ---------------------------------------------------------------------------
# Comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bar_chart(metrics_dict: dict) -> None:
    """
    Grouped bar chart — all models vs all four metrics.

    Each metric group shows one bar per model side by side.
    Value labels are printed on top of each bar.
    Saved to results/model_comparison_chart.png at 300 DPI.

    This is Figure 1 (or equivalent) in the Results section of your paper.
    """
    present_models = [m for m in MODEL_ORDER if m in metrics_dict]
    n_metrics = len(METRIC_ORDER)
    n_models  = len(present_models)
    bar_width = 0.8 / n_models
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model in enumerate(present_models):
        scores = metrics_dict[model]
        values = [scores.get(m, 0.0) for m in METRIC_ORDER]
        offsets = x + i * bar_width - (n_models - 1) * bar_width / 2
        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            label=model,
            color=COLOURS[i % len(COLOURS)],
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Model Comparison — Hinglish Sentiment Analysis",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylim(0, 1.08)
    ax.legend(title="Model", fontsize=10, title_fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "model_comparison_chart.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot_comparison_bar_chart] Saved to '{save_path}'.")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def generate_results_table(metrics_dict: dict) -> pd.DataFrame:
    """
    Print a clean comparison table and save it as CSV.

    Output: results/results_table.csv
    Copy the printed table directly into your paper's Results section.
    """
    rows = []
    for model in MODEL_ORDER:
        if model not in metrics_dict:
            continue
        scores = metrics_dict[model]
        rows.append({
            "Model":              model,
            "Accuracy":           scores.get("accuracy", float("nan")),
            "F1 (Macro)":         scores.get("f1_macro", float("nan")),
            "Precision (Macro)":  scores.get("precision_macro", float("nan")),
            "Recall (Macro)":     scores.get("recall_macro", float("nan")),
        })

    df = pd.DataFrame(rows).set_index("Model")
    df = df.map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)

    print("\n" + "=" * 65)
    print("  Results Table")
    print("=" * 65)
    print(df.to_string())
    print("=" * 65)

    csv_path = os.path.join(RESULTS_DIR, "results_table.csv")
    df.to_csv(csv_path)
    print(f"\n[generate_results_table] Saved to '{csv_path}'.")
    return df


# ---------------------------------------------------------------------------
# Combined confusion matrix figure
# ---------------------------------------------------------------------------

def plot_all_confusion_matrices(metrics_dict: dict) -> None:
    """
    Arrange all four confusion matrix PNGs into a 2×2 subplot grid.

    This single figure is commonly included in NLP papers to compare
    per-class performance across models at a glance.

    Saved to results/all_confusion_matrices.png.
    """
    present_models = [m for m in MODEL_ORDER if m in metrics_dict]
    n = len(present_models)
    if n == 0:
        print("[plot_all_confusion_matrices] No models found. Skipping.")
        return

    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = np.array(axes).flatten()

    for i, model in enumerate(present_models):
        safe_name = model.replace(" ", "_")
        img_path  = os.path.join(RESULTS_DIR, f"confusion_matrix_{safe_name}.png")

        if not os.path.exists(img_path):
            axes[i].set_title(f"{model}\n(not found)", fontsize=11)
            axes[i].axis("off")
            continue

        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(model, fontsize=12, fontweight="bold")
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Confusion Matrices — All Models",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "all_confusion_matrices.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot_all_confusion_matrices] Saved to '{save_path}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metrics = load_all_metrics()
    plot_comparison_bar_chart(metrics)
    generate_results_table(metrics)
    plot_all_confusion_matrices(metrics)
    print("\n[evaluate.py] All figures and tables generated. Check results/")
