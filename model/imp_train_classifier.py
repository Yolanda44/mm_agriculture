"""
Improved supervised topic classifier for Myanmar Agri-Assist.

Upgraded features:
- Uses sentence-transformer embeddings (semantic understanding)
- Cleans and normalizes category labels
- Removes categories with too few samples (<10)
- Oversamples minority classes for balanced training
- Saves classifier + label encoder + model name
- Pickle-safe (no lambdas)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from sentence_transformers import SentenceTransformer


# -------------------------
# CONFIG
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Agriculture_cleaned.csv" 
MODEL_PATH = BASE_DIR / "model" / "topic_model_v2.pkl"
REPORT_PATH = BASE_DIR / "model" / "classification_report_v2.txt"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# -------------------------
# LOAD + CLEAN DATA
# -------------------------
def load_dataset(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    df = df.rename(columns=str.strip)

    required = ["Instruction", "Output", "Category"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from dataset.")

    df["Instruction"] = df["Instruction"].fillna("").astype(str)
    df["Output"] = df["Output"].fillna("").astype(str)
    df["Category"] = df["Category"].fillna("Unknown").astype(str)

    # Clean category labels
    df["Category"] = (
        df["Category"]
        .str.replace(r"^\.+", "", regex=True)    # remove leading dots
        .str.replace(r"\s+", " ", regex=True)   # collapse whitespace
        .str.strip()
    )

    # Combine fields
    df["combined_text"] = (df["Instruction"] + " " + df["Output"]).str.strip()

    return df


# -------------------------
# FILTER + BALANCE DATA
# -------------------------
def filter_and_balance(df: pd.DataFrame, min_samples: int = 10) -> pd.DataFrame:
    counts = df["Category"].value_counts()
    valid = counts[counts >= min_samples].index

    # Remove tiny classes
    df = df[df["Category"].isin(valid)]

    # Oversampling to balance
    groups = []
    target_size = counts[counts >= min_samples].max()

    for cat, g in df.groupby("Category"):
        if len(g) < target_size:
            g = resample(g, replace=True, n_samples=target_size, random_state=42)
        groups.append(g)

    return pd.concat(groups).reset_index(drop=True)


# -------------------------
# EMBEDDING + TRAINING
# -------------------------
def train_classifier(df: pd.DataFrame) -> Dict[str, Any]:
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Encode texts
    print("Embedding texts...")
    embeddings = model.encode(
        df["combined_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    y = df["Category"].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    bundle = {
        "classifier": clf,
        "label_encoder": label_encoder,
        "embedding_model_name": EMBED_MODEL_NAME,
    }
    return bundle, report


# -------------------------
# MAIN
# -------------------------
def main():
    df = load_dataset(DATA_PATH)
    print("Loaded:", df.shape)

    df = filter_and_balance(df)
    print("Balanced dataset:", Counter(df["Category"]))

    bundle, report = train_classifier(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\nSaved improved classifier to:", MODEL_PATH)
    print("\nClassification Report:\n")
    print(report)


if __name__ == "__main__":
    main()
