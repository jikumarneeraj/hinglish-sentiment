"""
preprocess.py
-------------
Step 1 of the pipeline. Loads the raw SemEval 2020 Hinglish dataset,
cleans each tweet, encodes labels, and saves a clean CSV to data/processed/.

Run:
    python src/preprocess.py
"""

import os
import re
import pandas as pd


# ---------------------------------------------------------------------------
# Label mapping — SemEval 2020 Task 9 uses these three classes
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
}


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV dataset and return a DataFrame with
    standardised column names: 'text' and 'label'.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file (e.g. data/raw/hinglish_train.csv).

    Returns
    -------
    pd.DataFrame
        DataFrame with at least 'text' and 'label' columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download SemEval 2020 Task 9 from "
            "https://ritual.uh.edu/semeval-2020-task9/ "
            "and place the CSV at data/raw/hinglish_train.csv"
        )

    df = pd.read_csv(filepath)

    # Normalise common alternate column names
    rename_map = {}
    if "tweet" in df.columns and "text" not in df.columns:
        rename_map["tweet"] = "text"
    if "sentiment" in df.columns and "label" not in df.columns:
        rename_map["sentiment"] = "label"
    if rename_map:
        df = df.rename(columns=rename_map)

    assert "text" in df.columns, (
        "Could not find a 'text' or 'tweet' column. "
        f"Available columns: {list(df.columns)}"
    )
    assert "label" in df.columns, (
        "Could not find a 'label' or 'sentiment' column. "
        f"Available columns: {list(df.columns)}"
    )

    print(f"[load_data] Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
    print(df[["text", "label"]].head())
    return df[["text", "label"]]


def clean_text(text: str) -> str:
    """
    Clean a single raw tweet string.

    Cleaning steps (applied in order):
        1. Lowercase
        2. Remove URLs (http/https/www)
        3. Remove @mentions
        4. Remove '#' symbol but keep the word (e.g. #cricket → cricket)
        5. Remove characters that are not letters, digits, spaces,
           apostrophes, or Devanagari Unicode (U+0900–U+097F)
        6. Collapse multiple spaces and strip leading/trailing whitespace

    NOTE: Devanagari characters are intentionally preserved — they are
    valid linguistic content, not noise.

    Parameters
    ----------
    text : str
        Raw tweet string.

    Returns
    -------
    str
        Cleaned tweet string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 4. Remove '#' but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # 5. Keep: ASCII letters/digits, spaces, apostrophes, Devanagari
    #    Remove everything else (punctuation, emojis, special symbols)
    text = re.sub(r"[^a-z0-9 '\u0900-\u097F]", " ", text)

    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text() to every row and encode labels as integers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with 'text' (str) and 'label' (int 0/1/2).
    """
    # Apply text cleaning
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)

    # Drop rows that became empty after cleaning
    before = len(df)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[preprocess_dataset] Dropped {before - after} empty rows after cleaning.")

    # Encode string labels → integers
    df["label_str"] = df["label"].astype(str).str.lower().str.strip()
    df["label"] = df["label_str"].map(LABEL_MAP)

    unmapped = df["label"].isna().sum()
    if unmapped > 0:
        unique_raw = df.loc[df["label"].isna(), "label_str"].unique()
        raise ValueError(
            f"{unmapped} rows have unrecognised label values: {unique_raw}. "
            f"Expected one of: {list(LABEL_MAP.keys())}"
        )

    df["label"] = df["label"].astype(int)
    df = df[["text", "label"]]

    print(f"[preprocess_dataset] Dataset ready: {len(df)} rows.")
    print(df["label"].value_counts().rename({0: "positive", 1: "negative", 2: "neutral"}))
    return df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    output_path : str
        Destination path (e.g. data/processed/clean_data.csv).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[save_processed] Saved {len(df)} rows to '{output_path}'.")


# ---------------------------------------------------------------------------
# Entry point — run this file directly to execute the full pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    RAW_PATH = "data/raw/hinglish_train.csv"
    PROCESSED_PATH = "data/processed/clean_data.csv"

    df = load_data(RAW_PATH)
    df = preprocess_dataset(df)
    save_processed(df, PROCESSED_PATH)
