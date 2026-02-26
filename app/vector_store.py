"""
vector_store.py – Semantic Search Engine for scAlerite
======================================================

🧠 HOW FAISS WORKS:
───────────────────
FAISS (Facebook AI Similarity Search) finds vectors that are
mathematically closest to a query vector. We use IndexFlatL2
(Euclidean distance). Lower score = better match.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # macOS threading fix

import faiss
import numpy as np
from app.embedding import get_embeddings, embed_query


class VectorStore:
    """In-memory vector store using FAISS with source tracking."""

    def __init__(self, dimension: int = 384):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []    # text content
        self.sources = []   # source PDF filename for each chunk

    def add_chunks(self, chunk_dicts: list[dict]):
        """
        Add chunks with metadata to the index.

        Args:
            chunk_dicts: List of dicts with 'text' and 'source' keys.
        """
        if not chunk_dicts:
            return

        texts = [c["text"] for c in chunk_dicts]
        sources = [c["source"] for c in chunk_dicts]

        print(f"🧠 Generating embeddings for {len(texts)} chunks...")
        embeddings = get_embeddings(texts).astype('float32')

        self.index.add(embeddings)
        self.chunks.extend(texts)
        self.sources.extend(sources)
        print(f"✅ Added {len(texts)} chunks to FAISS index.")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for the most relevant chunks.

        Returns:
            List of dicts with 'chunk', 'source', and 'score'.
        """
        if self.index.ntotal == 0:
            return []

        query_vec = embed_query(query).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "source": self.sources[idx],
                    "score": float(dist),
                })

        return results


if __name__ == "__main__":
    store = VectorStore()
    store.add_chunks([
        {"text": "The academic calendar starts in August 2025.", "source": "Calendar.pdf"},
        {"text": "Students must book meeting rooms 24 hours in advance.", "source": "SOP.pdf"},
    ])

    matches = store.search("When do classes begin?", top_k=2)
    for i, m in enumerate(matches, 1):
        print(f"Match {i} (Score: {m['score']:.4f}, Source: {m['source']}): {m['chunk']}")
