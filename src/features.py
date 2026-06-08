"""
features.py
-----------
Step 2 of the pipeline. Converts cleaned text into numerical
representations for each model type:
    - TF-IDF sparse vectors  →  Logistic Regression & SVM
    - Padded integer sequences  →  LSTM
    - HuggingFace tokenizer  →  MuRIL transformer

IMPORTANT: All four models use the EXACT same train/val/test split
(random_state=42, ratio 80/10/10). Do not change this.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_clean_data(filepath: str = "data/processed/clean_data.csv"):
    """
    Load the preprocessed CSV produced by preprocess.py.

    Returns
    -------
    texts : list[str]
    labels : list[int]
    """
    df = pd.read_csv(filepath)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    print(f"[load_clean_data] Loaded {len(texts)} samples.")
    return texts, labels


def split_data(texts, labels):
    """
    Split into train (80%), validation (10%), test (10%).

    random_state=42 ensures every model sees the EXACT same split —
    this is required for a valid model comparison.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    # Second split: 50/50 of temp → 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(
        f"[split_data] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_tfidf_features(X_train, X_val, X_test):
    """
    Fit a TF-IDF vectorizer on training data only, then transform all splits.

    Hyperparameters:
        max_features = 50,000   (vocabulary size cap)
        ngram_range  = (1, 2)   (unigrams + bigrams)

    Saves fitted vectorizer to results/tfidf_vectorizer.pkl for inference.

    Returns
    -------
    train_tfidf, val_tfidf, test_tfidf : scipy sparse matrices
    """
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))

    # Fit ONLY on training data to avoid data leakage
    train_tfidf = vectorizer.fit_transform(X_train)
    val_tfidf = vectorizer.transform(X_val)
    test_tfidf = vectorizer.transform(X_test)

    # Save fitted vectorizer
    vectorizer_path = os.path.join(RESULTS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"[get_tfidf_features] Vectorizer saved to '{vectorizer_path}'.")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    return train_tfidf, val_tfidf, test_tfidf


def get_lstm_sequences(
    X_train, X_val, X_test, max_words: int = 30_000, max_len: int = 100
):
    """
    Convert text to padded integer sequences for the LSTM model.

    Hyperparameters:
        max_words = 30,000   (vocabulary size for Keras Tokenizer)
        max_len   = 100      (sequence length — tweets are typically short)

    Fit ONLY on X_train to avoid data leakage.
    Saves fitted tokenizer to results/lstm_tokenizer.pkl.

    Returns
    -------
    train_seq, val_seq, test_seq : np.ndarray of shape (n, max_len)
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)  # fit on training data only

    def _to_padded(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

    train_seq = _to_padded(X_train)
    val_seq = _to_padded(X_val)
    test_seq = _to_padded(X_test)

    tokenizer_path = os.path.join(RESULTS_DIR, "lstm_tokenizer.pkl")
    joblib.dump(tokenizer, tokenizer_path)
    print(f"[get_lstm_sequences] Tokenizer saved to '{tokenizer_path}'.")
    print(f"  Vocab size: {len(tokenizer.word_index)} | Sequence length: {max_len}")

    return train_seq, val_seq, test_seq


def get_transformer_tokenizer(model_name: str = "google/muril-base-cased"):
    """
    Load and return the HuggingFace tokenizer for MuRIL.

    The transformer model handles its own tokenization internally;
    this function simply loads the tokenizer for use in transformer_model.py.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: google/muril-base-cased.

    Returns
    -------
    AutoTokenizer
    """
    from transformers import AutoTokenizer

    print(f"[get_transformer_tokenizer] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[get_transformer_tokenizer] Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    return tokenizer
