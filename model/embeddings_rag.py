"""
Myanmar Agri-Assist Retrieval Augmented Generation utilities.

Creates a FAISS embeddings index using multilingual sentence-transformers while
ensuring Burmese-aware tokenization via pyidaungsu. The resulting RAGPipeline
object exposes helper methods to retrieve top-k passages and craft a templated
answer for FastAPI and Streamlit layers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import pandas as pd
from mmdt_tokenizer import MyanmarTokenizer
from sentence_transformers import SentenceTransformer



tokenizer = MyanmarTokenizer()
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Agriculture.csv"
INDEX_PATH = BASE_DIR / "model" / "faiss_index.bin"
METADATA_PATH = BASE_DIR / "model" / "faiss_metadata.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def burmese_tokenizer(text: str) -> list:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    tokens = tokenizer.word_tokenize(text)

    # unwrap nested list from mmdt-tokenizer
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]

    return [tok.strip().lower() for tok in tokens if isinstance(tok, str) and tok.strip()]

def preprocess_text(text: str) -> str:
    return " ".join(burmese_tokenizer(text))


class RAGPipeline:
    def __init__(
        self,
        data_path: Path = DATA_PATH,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        model_name: str = MODEL_NAME,
    ):
        self.data_path = data_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata: List[Dict[str, str]] = []
        self._load_or_build()

    def _load_or_build(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return
        self.build_index()

    def build_index(self) -> None:
        df = pd.read_csv(self.data_path)
        df = df.rename(columns=str.strip)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df["Instruction"] = df["Instruction"].fillna("").astype(str)
        df["Output"] = df["Output"].fillna("").astype(str)
        df["Category"] = df["Category"].fillna("Unknown").astype(str)
        combined_corpus: List[str] = []
        metadata: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            text = preprocess_text(f"{row['Instruction']} {row['Output']}")
            combined_corpus.append(text)
            metadata.append(
                {
                    "category": row["Category"],
                    "instruction": row["Instruction"],
                    "output": row["Output"],
                }
            )
        embeddings = self.model.encode(
            combined_corpus,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        self.metadata = metadata
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _ensure_index(self) -> None:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

    def retrieve_top_k(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        self._ensure_index()
        query_text = preprocess_text(query)
        query_vec = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_vec, k)
        contexts: List[Dict[str, str]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx].copy()
            entry["score"] = float(score)
            contexts.append(entry)
        return contexts

    def generate_answer(self, query: str, contexts: List[Dict[str, str]]) -> str:
        if not contexts:
            return (
                "No relevant agricultural knowledge was retrieved. "
                "Please provide additional Burmese details so the assistant can respond."
            )
        summary_lines = ["Top knowledge hits that support your Burmese question:"]
        for idx, ctx in enumerate(contexts, start=1):
            summary_lines.append(f"{idx}. ({ctx['category']}) {ctx['output']}")
        summary_lines.append(
            "Combine these agronomic recommendations with local field knowledge "
            "to guide farmers effectively."
        )
        return "\n".join(summary_lines)


if __name__ == "__main__":
    rag = RAGPipeline()
    print("RAG index ready. Metadata records:", len(rag.metadata))
