"""
Supervised topic classification training script for Myanmar Agri-Assist.

This module trains a Logistic Regression classifier using TF-IDF features built
with a Burmese tokenizer from pyidaungsu. The trained artefacts (model,
vectorizer, and label encoder) are saved together in topic_model.pkl so that
both the API backend and Streamlit UI can perform predictions consistently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from mmdt_tokenizer import MyanmarTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Agriculture.csv"
MODEL_PATH = BASE_DIR / "model" / "topic_model.pkl"
REPORT_PATH = BASE_DIR / "model" / "classification_report.txt"
tokenizer = MyanmarTokenizer()

def burmese_tokenizer(text: str) -> list:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    tokens = tokenizer.word_tokenize(text)

    # unwrap nested list from mmdt-tokenizer
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]

    return [tok.strip().lower() for tok in tokens if isinstance(tok, str) and tok.strip()]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=str.strip)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["Instruction"] = df["Instruction"].fillna("").astype(str)
    df["Output"] = df["Output"].fillna("").astype(str)
    df["Category"] = df["Category"].fillna("Unknown").astype(str)
    df["combined_text"] = (df["Instruction"] + " " + df["Output"]).str.strip()
    return df


def identity(x):
    return x

def train_classifier(df: pd.DataFrame) -> Tuple[Dict[str, object], str]:
    X = df["combined_text"].values
    y = df["Category"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    vectorizer = TfidfVectorizer(
        tokenizer=burmese_tokenizer,
        preprocessor=identity,
        lowercase=False,
        token_pattern=None,
        max_features=10000,
    )


    stratify_param = y_encoded if pd.Series(y).value_counts().min() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=stratify_param
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifier = LogisticRegression(max_iter=1000, multi_class="auto")
    classifier.fit(X_train_vec, y_train)

    y_pred = classifier.predict(X_test_vec)
    num_classes = len(label_encoder.classes_)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(num_classes)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    bundle = {
        "model": classifier,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
    }
    return bundle, report


def main() -> None:
    df = load_dataset()
    bundle, report = train_classifier(df)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved model bundle to {MODEL_PATH}")
    print(f"Classification report:\n{report}")


if __name__ == "__main__":
    main()
