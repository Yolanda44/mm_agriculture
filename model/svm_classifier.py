"""
Optimized SVM Topic Classifier (Myanmar Agri-Assist)

- Uses SentenceTransformer embeddings
- Cleans & merges categories
- Removes rare categories
- Oversamples smaller categories
- Uses LinearSVC (strongest baseline for text)
- Supports probability output (optional)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from collections import Counter

import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sentence_transformers import SentenceTransformer


# -------------------------------
# CONFIG
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Agriculture_cleaned.csv"   # <-- your cleaned dataset
MODEL_PATH = BASE_DIR / "model" / "topic_model_svm.pkl"
REPORT_PATH = BASE_DIR / "model" / "classification_report_svm.txt"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# -------------------------------
# LOAD + CLEAN
# -------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=str.strip)

    df["Instruction"] = df["Instruction"].fillna("").astype(str)
    df["Output"] = df["Output"].fillna("").astype(str)
    df["Category"] = df["Category"].fillna("Unknown").astype(str)

    # Normalize categories
    df["Category"] = (
        df["Category"]
        .str.replace(r"^\.+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["combined_text"] = (df["Instruction"] + " " + df["Output"]).str.strip()
    return df


# -------------------------------
# BALANCE SMALL CATEGORIES
# -------------------------------
def balance_dataset(df: pd.DataFrame, min_samples=40) -> pd.DataFrame:
    counts = df["Category"].value_counts()

    # filter out super-small categories
    valid_categories = counts[counts >= min_samples].index
    df = df[df["Category"].isin(valid_categories)]

    # Oversample smaller groups
    target_n = counts[counts >= min_samples].max()

    balanced = []
    for cat, group in df.groupby("Category"):
        if len(group) < target_n:
            group = resample(group, replace=True, n_samples=target_n, random_state=42)
        balanced.append(group)

    return pd.concat(balanced).reset_index(drop=True)


# -------------------------------
# TRAIN SVM
# -------------------------------
def train_svm(df: pd.DataFrame):
    # Load embedding model
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("🔄 Encoding sentences...")
    X_embeddings = embed_model.encode(
        df["combined_text"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    y = df["Category"].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

    print("🔄 Training SVM classifier...")
    svm = LinearSVC()

    # OPTIONAL: probability calibration (helps for confidence scores)
    clf = CalibratedClassifierCV(svm, cv=3)

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


# -------------------------------
# MAIN
# -------------------------------
def main():
    df = load_dataset(DATA_PATH)
    print("Loaded dataset:", df.shape)

    df = balance_dataset(df)   # Balanced training
    print("Balanced counts:\n", df["Category"].value_counts())

    bundle, report = train_svm(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\n🎉 Saved SVM model:", MODEL_PATH)
    print("\n📊 Classification Report:\n")
    print(report)


if __name__ == "__main__":
    main()
