"""
Myanmar Agri-Assist - Exploratory Data Analysis module.

This script loads the Agriculture dataset, tokenizes the Burmese text with
pyidaungsu, creates distribution charts, and extracts top TF-IDF keywords per
category. Outputs are stored inside the eda/ directory so that downstream
components (Streamlit UI) can re-use the generated artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import pyidaungsu as pds
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Agriculture.csv"
OUTPUT_DIR = BASE_DIR / "eda"
PIE_PATH = OUTPUT_DIR / "category_pie.png"
BAR_PATH = OUTPUT_DIR / "category_bar.png"
KEYWORDS_PATH = OUTPUT_DIR / "category_keywords.json"


def burmese_tokenizer(text: str) -> List[str]:
    """
    Tokenize Burmese text using pyidaungsu's word tokenizer. The TF-IDF vectorizer
    expects lowercase string tokens separated by whitespace, so we normalize here.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    tokens = pds.tokenize(text, form="word")
    return [tok.strip().lower() for tok in tokens if tok.strip()]


def clean_and_join(text: str) -> str:
    """Tokenize and join Burmese text back into a whitespace separated string."""
    return " ".join(burmese_tokenizer(text))


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=str.strip)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    for column in ("Instruction", "Output", "Category"):
        if column not in df.columns:
            raise ValueError(f"Missing '{column}' column in dataset.")
    df["Instruction"] = df["Instruction"].fillna("").astype(str)
    df["Output"] = df["Output"].fillna("").astype(str)
    df["Category"] = df["Category"].fillna("Unknown").astype(str)
    df["combined_text"] = (df["Instruction"] + " " + df["Output"]).map(clean_and_join)
    return df


def plot_category_pie(df: pd.DataFrame) -> None:
    distribution = df["Category"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(
        distribution.values,
        labels=distribution.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Category Distribution (Pie)")
    plt.tight_layout()
    fig.savefig(PIE_PATH, dpi=300)
    plt.close(fig)


def plot_category_bar(df: pd.DataFrame, top_n: int = 15) -> None:
    distribution = df["Category"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(12, 6))
    distribution.plot(kind="bar", ax=ax, color="#1b9e77")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Top {top_n} Categories")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(BAR_PATH, dpi=300)
    plt.close(fig)


def extract_keywords(df: pd.DataFrame, top_n: int = 10) -> Dict[str, List[str]]:
    """
    Extract keywords per category using TF-IDF. We compute the mean TF-IDF
    score of each term inside a category subset and keep the top scoring tokens.
    """
    vectorizer = TfidfVectorizer(
        tokenizer=burmese_tokenizer,
        preprocessor=lambda text: text,
        token_pattern=None,
        lowercase=False,
        max_features=5000,
    )
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    feature_names = vectorizer.get_feature_names_out()

    keyword_map: Dict[str, List[str]] = {}
    for category in df["Category"].unique():
        mask = df["Category"] == category
        if mask.sum() == 0:
            continue
        category_matrix = tfidf_matrix[mask]
        mean_scores = category_matrix.mean(axis=0).A1
        top_indices = mean_scores.argsort()[::-1][:top_n]
        keyword_map[category] = [feature_names[idx] for idx in top_indices]

    return keyword_map


def save_keywords(keywords: Dict[str, Iterable[str]]) -> None:
    KEYWORDS_PATH.write_text(
        json.dumps(keywords, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_eda() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    plot_category_pie(df)
    plot_category_bar(df)
    keywords = extract_keywords(df)
    save_keywords(keywords)
    print(f"Saved pie chart to {PIE_PATH}")
    print(f"Saved bar chart to {BAR_PATH}")
    print(f"Saved keyword summary to {KEYWORDS_PATH}")


if __name__ == "__main__":
    run_eda()
