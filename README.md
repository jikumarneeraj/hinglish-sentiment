# Hinglish Sentiment Analysis

**Comparative Analysis of ML and Transformer Models for Hinglish Sentiment Analysis**

B.Tech Research Paper | SemEval 2020 Task 9 Dataset

---

## Overview

This project benchmarks four models for sentiment analysis on Hinglish (Hindi-English code-mixed) text:

| Model | Type | Library |
|---|---|---|
| Logistic Regression | Classical ML | scikit-learn |
| SVM (LinearSVC) | Classical ML | scikit-learn |
| Bidirectional LSTM | Deep Learning | TensorFlow/Keras |
| MuRIL | Transformer | HuggingFace |

---

## Project Structure

```
hinglish-sentiment/
├── data/
│   ├── raw/                  ← Place downloaded SemEval 2020 dataset here
│   └── processed/            ← Cleaned data saved here by preprocess.py
├── notebooks/
│   └── EDA.ipynb             ← Exploratory data analysis
├── src/
│   ├── preprocess.py         ← Step 1: Clean raw text
│   ├── features.py           ← Step 2: Text → numerical features
│   ├── train.py              ← Step 3: Unified training entry point
│   ├── evaluate.py           ← Step 4: Metrics + comparison charts
│   └── models/
│       ├── __init__.py
│       ├── classical.py      ← Logistic Regression + SVM
│       ├── lstm_model.py     ← Bidirectional LSTM
│       └── transformer_model.py  ← MuRIL fine-tuning
├── results/                  ← Metrics, plots, checkpoints (auto-generated)
├── paper/                    ← LaTeX/Word paper draft
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/hinglish-sentiment.git
cd hinglish-sentiment
pip install -r requirements.txt
```

### 2. Download the dataset

Download the SemEval 2020 Task 9 Hinglish Sentiment dataset:
- URL: https://ritual.uh.edu/semeval-2020-task9/
- Place the CSV file at: `data/raw/hinglish_train.csv`

Expected columns: `text` (or `tweet`) and `label` (or `sentiment`)

---

## Run Order

Run all steps in this exact order:

```bash
# Step 1 — Clean and preprocess raw data
python src/preprocess.py

# Step 2 — Run EDA notebook (Jupyter)
jupyter notebook notebooks/EDA.ipynb

# Step 3 — Train classical models
python src/train.py --model lr
python src/train.py --model svm

# Step 4 — Train LSTM (takes 10–20 minutes on CPU)
python src/train.py --model lstm

# Step 5 — Fine-tune MuRIL (use Google Colab with GPU recommended)
python src/train.py --model muril

# Step 6 — Generate comparison figures and tables
python src/evaluate.py

# Step 7 — Extract misclassified examples for error analysis (after MuRIL)
python src/error_analysis.py

# Alternatively, train all models at once:
python src/train.py --model all
```

---

## Google Colab (for MuRIL fine-tuning)

MuRIL fine-tuning requires a GPU. Use the included Colab notebook:

1. Upload `hinglish-sentiment-colab.zip` or open `notebooks/MuRIL_Colab.ipynb`
2. Runtime → Change runtime type → GPU (T4)
3. Install with `pip install -r requirements-colab.txt` (**not** `requirements.txt`)
4. Run: `python src/train.py --model muril`

Expected time: 20–30 minutes on T4 GPU.

---

## Results

### Final benchmark (test set, 1,513 samples)

| Model | Accuracy | F1 (Macro) | Precision | Recall |
|-------|----------|------------|-----------|--------|
| Logistic Regression | 0.6219 | 0.6246 | 0.6211 | 0.6310 |
| SVM | 0.6114 | 0.6130 | 0.6098 | 0.6204 |
| LSTM | 0.6008 | 0.6004 | 0.5970 | 0.6110 |
| **MuRIL** | **0.6418** | **0.6452** | **0.6528** | **0.6408** |

MuRIL outperforms all baselines by **+2.1% F1** over the best classical model (Logistic Regression).

All outputs are saved to `results/`:

| File | Description |
|---|---|
| `results/metrics.json` | All model metrics (accuracy, F1, precision, recall) |
| `results/model_comparison_chart.png` | Grouped bar chart — all models vs all metrics |
| `results/confusion_matrix_*.png` | Per-model confusion matrices |
| `results/all_confusion_matrices.png` | 2×2 grid of all confusion matrices |
| `results/results_table.csv` | Comparison table for paper |
| `results/lstm_training_curves.png` | LSTM training/validation curves |
| `results/muril_checkpoints/checkpoint-3785/` | Best MuRIL fine-tuned weights |
| `results/error_analysis.csv` | Misclassified test examples for paper |
| `results/error_analysis.txt` | 15 sample errors for Error Analysis section |

---

## Reproducibility

All random seeds are fixed to `42` across:
- Python `random`
- NumPy
- PyTorch
- TensorFlow/Keras

The train/val/test split (80/10/10) uses `random_state=42` and is identical across all four models.

---

## Dataset

**SemEval 2020 Task 9 — Sentiment Analysis for Code-Mixed Social Media Text**

- ~15,130 labeled tweets
- 3 classes: Positive, Negative, Neutral
- Language: Hinglish (Hindi-English code-mixed, Roman script + Devanagari)
- Source: Twitter

---

## Target Venues

- IEEE INDICON
- ICCES (International Conference on Computer Engineering and Systems)
- Elsevier Expert Systems with Applications

---

## Citation

If you use this code, please cite:

```
@article{your_paper_2024,
  title={Comparative Analysis of ML and Transformer Models for Hinglish Sentiment Analysis},
  author={Neeraj Kumar},
  year={2026}
}
```

Also cite the dataset:
```
@inproceedings{patwa2020semeval,
  title={SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text},
  author={Patwa, Parth and others},
  booktitle={Proceedings of the 14th International Workshop on Semantic Evaluation},
  year={2020}
}
```

---

*Work honestly. Report all results. The publication will follow.*
