"""
XGBoost-powered topic classifier for Myanmar Agri-Assist.

Features:
- Uses SentenceTransformer embeddings
- Cleans + merges categories
- Removes rare classes
- Balances dataset (oversampling)
- Trains XGBoost multi-class classifier
- Saves model bundle (pickle-safe)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from sentence_transformers import SentenceTransformer
import xgboost as xgb


# ------------------------------------
# CONFIG
# ------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Agriculture_cleaned.csv"   # <-- use your final cleaned file
MODEL_PATH = BASE_DIR / "model" / "topic_model_xgb.pkl"
REPORT_PATH = BASE_DIR / "model" / "classification_report_xgb.txt"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ------------------------------------
# LOAD + CLEAN
# ------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=str.strip)

    df["Instruction"] = df["Instruction"].fillna("").astype(str)
    df["Output"] = df["Output"].fillna("").astype(str)
    df["Category"] = df["Category"].fillna("Unknown").astype(str)

    # Clean labels
    df["Category"] = (
        df["Category"]
        .str.replace(r"^\.+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["combined_text"] = (df["Instruction"] + " " + df["Output"]).str.strip()
    return df


# ------------------------------------
# BALANCE DATASET
# ------------------------------------
def balance_dataset(df: pd.DataFrame, min_samples=40) -> pd.DataFrame:
    counts = df["Category"].value_counts()
    valid = counts[counts >= min_samples].index
    df = df[df["Category"].isin(valid)]

    target_n = counts[counts >= min_samples].max()

    balanced = []
    for cat, group in df.groupby("Category"):
        if len(group) < target_n:
            group = resample(group, replace=True, n_samples=target_n, random_state=42)
        balanced.append(group)

    return pd.concat(balanced).reset_index(drop=True)


# ------------------------------------
# TRAIN XGBOOST CLASSIFIER
# ------------------------------------
def train_xgboost(df: pd.DataFrame):

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("🔄 Encoding sentences...")
    X = embed_model.encode(
        df["combined_text"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    y = df["Category"].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    print("🔄 Training XGBoost classifier...")
    clf = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        learning_rate=0.05,
        n_estimators=300,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",       # fast CPU training
        random_state=42,
    )

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


# ------------------------------------
# MAIN
# ------------------------------------
def main():
    df = load_dataset(DATA_PATH)
    print("Loaded:", df.shape)

    df = balance_dataset(df)
    print("Balanced dataset:\n", df["Category"].value_counts())

    bundle, report = train_xgboost(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\n🎉 Saved XGBoost model:", MODEL_PATH)
    print("\n📊 Classification Report:\n")
    print(report)


if __name__ == "__main__":
    main()
