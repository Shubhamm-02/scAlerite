"""
vector_store.py – Lightweight TF-IDF based semantic search
============================================================

Replaces FAISS + sentence-transformers (400MB RAM) with scikit-learn
TF-IDF (30MB RAM) — fits comfortably inside Render's 512MB free tier.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class VectorStore:
    def __init__(self):
        self.chunks = []        # list of {"text": ..., "source": ...}
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),   # unigrams + bigrams for better coverage
        )
        self._matrix = None     # TF-IDF matrix (n_docs x n_features)

    def add_chunks(self, chunk_dicts: list[dict]):
        """
        Build a TF-IDF matrix from all chunks.
        chunk_dicts: [{"text": ..., "source": ...}, ...]
        """
        self.chunks = chunk_dicts
        texts = [c["text"] for c in chunk_dicts]
        self._matrix = self.vectorizer.fit_transform(texts)
        print(f"✅ TF-IDF index built: {self._matrix.shape[0]} docs, {self._matrix.shape[1]} features")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Find top-k most relevant chunks for a query.
        Returns: [{"chunk": text, "source": filename, "score": float}, ...]
        """
        if self._matrix is None or len(self.chunks) == 0:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:   # only return relevant results
                results.append({
                    "chunk": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": float(scores[idx]),
                })
        return results
