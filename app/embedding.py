"""
embedding.py – Vector Generation for scAlerite
==============================================

🧠 THE REFINED EMBEDDING MODULE:
────────────────────────────────
This version is optimized for stability and specific requirements:
1. FORCE CPU: We explicitly set the device to 'cpu' as requested.
2. NUMPY OUTPUT: Standard format for vector databases like FAISS.
3. BATCHING: Efficiently handles large lists of text.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Singleton pattern for the model
_MODEL = None

def get_model():
    """
    Load the sentence-transformer model on CPU.
    """
    global _MODEL
    if _MODEL is None:
        print("🤖 Loading Embedding Model (all-MiniLM-L6-v2) on CPU...")
        # device='cpu' ensures we don't accidentally try to use MPS/CUDA
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("✅ Model Loaded.")
    return _MODEL

def get_embeddings(text_list: list[str]) -> np.ndarray:
    """
    Convert a list of strings into a numpy array of embeddings.

    🧠 NLP NOTE — Batching:
    ────────────────────────
    Processing 100 chunks one-by-one is slow. Processing them as a
    'batch' allows the transformer to use matrix operations effectively.
    sentence-transformers handles this internally when you pass a list.

    Args:
        text_list: List of text chunks.

    Returns:
        Numpy array of shape (len(text_list), 384).
    """
    if not text_list:
        return np.array([])

    model = get_model()
    # convert_to_numpy=True is the default, but we make it explicit
    embeddings = model.encode(
        text_list,
        batch_size=32,       # Standard batch size for CPU
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings

def embed_query(query: str) -> np.ndarray:
    """
    Convert a single query into a 1D numpy array.
    """
    model = get_model()
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding

if __name__ == "__main__":
    # Test batching and numpy output
    test_data = ["Hello scAlerite", "NLP is fun", "Batching works!"]
    results = get_embeddings(test_data)

    print(f"\n📊 Result Type: {type(results)}")
    print(f"📐 Result Shape: {results.shape}")
    print(f"🔢 First few values:\n{results[0][:5]}")
