from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import faiss
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]

XGB_MODEL_PATH = BASE_DIR / "model" / "topic_model_xgb.pkl"
FAISS_INDEX_PATH = BASE_DIR / "model" / "faiss_index.bin"
FAISS_METADATA_PATH = BASE_DIR / "model" / "faiss_metadata.json"

# Local model folder (recommended)
LOCAL_EMBED_MODEL = BASE_DIR / "model" / "paraphrase-multilingual-MiniLM-L12-v2"


class MyanmarAgriAssistEngine:

    def __init__(self):

        # ---------------------------------------
        # 1. Load sentence embedder FIRST
        # ---------------------------------------
        if LOCAL_EMBED_MODEL.exists():
            print("📌 Loading local embedding model:", LOCAL_EMBED_MODEL)
            self.embedder = SentenceTransformer(str(LOCAL_EMBED_MODEL))
        else:
            print("🌐 Local model not found — downloading from HuggingFace...")
            # Fallback: read embedding name later
            embed_model_name = joblib.load(XGB_MODEL_PATH)["embedding_model_name"]
            self.embedder = SentenceTransformer(embed_model_name)

        print("✅ Embedding model loaded.")

        # Lazy load the rest
        self._classifier = None
        self._label_encoder = None
        self._index = None
        self._metadata = None

    # ---------------------------------------
    # Lazy loading properties
    # ---------------------------------------
    @property
    def classifier(self):
        if self._classifier is None:
            print("📌 Loading classifier...")
            bundle = joblib.load(XGB_MODEL_PATH)
            self._classifier = bundle["classifier"]
            self._label_encoder = bundle["label_encoder"]
        return self._classifier

    @property
    def label_encoder(self):
        if self._label_encoder is None:
            _ = self.classifier     # loads both
        return self._label_encoder

    @property
    def index(self):
        if self._index is None:
            print("📌 Loading FAISS index...")
            self._index = faiss.read_index(str(FAISS_INDEX_PATH))
        return self._index

    @property
    def metadata(self):
        if self._metadata is None:
            print("📌 Loading metadata...")
            self._metadata = json.loads(
                FAISS_METADATA_PATH.read_text(encoding="utf-8")
            )
        return self._metadata

    # ---------------------------------------
    # Embedding helper
    # ---------------------------------------
    def embed(self, text: str) -> np.ndarray:
        emb = self.embedder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        return emb.astype("float32")

    # ---------------------------------------
    # Classification
    # ---------------------------------------
    def classify_text(self, text: str) -> Dict[str, Any]:
        emb = self.embed(text)
        pred_id = self.classifier.predict(emb)[0]

        # get confidence
        try:
            probas = self.classifier.predict_proba(emb)[0]
            confidence = float(np.max(probas))
        except Exception:
            confidence = 1.0

        category = self.label_encoder.inverse_transform([pred_id])[0]

        return {
            "category": category,
            "confidence": confidence,
        }

    # ---------------------------------------
    # RAG Retrieval
    # ---------------------------------------
    def retrieve_top_k(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self.embed(query)

        scores, indices = self.index.search(query_vec, k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue

            entry = self.metadata[idx].copy()
            entry["score"] = float(score)
            results.append(entry)

        return results

    # ---------------------------------------
    # RAG Answer
    # ---------------------------------------
    def ask_question(self, query: str, top_k: int = 3):
        contexts = self.retrieve_top_k(query, k=top_k)

        if not contexts:
            return {
                "answer": "လူကြီးမင်း၏ မေးခွန်းအတွက် သက်ဆိုင်ရာ အချက်အလက်များ မတွေ့ရှိနိုင်ပါ။",
                "contexts": [],
            }

        lines = ["Myanmar Agri-Assistမှအကြံပြုချက်များ"]
        for i, ctx in enumerate(contexts, start=1):
            lines.append(f"{i}. ({ctx['category']}) {ctx['output']}")

        lines.append("ဤအချက်အလက်များအပေါ် အခြေခံပြီး စိုက်ပျိုးရေးအကြံပြုချက်များ ရရှိနိုင်သည်။")

        return {
            "answer": "\n".join(lines),
            "contexts": contexts,
        }
