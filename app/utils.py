"""
utils.py – PDF Text Processing Pipeline for scAlerite
======================================================

This module is the FIRST stage of a RAG (Retrieval-Augmented Generation) pipeline.

🧠 WHY EACH STEP EXISTS:
─────────────────────────
1. EXTRACT  → LLMs can't read PDFs; we convert pages to plain text.
2. CLEAN    → Raw PDF text has broken lines, extra spaces, headers/footers.
              Cleaning normalizes it so the model isn't confused by noise.
3. CHUNK    → LLMs have a limited "context window" (the amount of text they
              can read at once). We split the document into small, focused
              pieces so we can later find the *most relevant* piece for a
              user's question and feed just that to the model.
4. OVERLAP  → If we split rigidly every 400 words, a sentence at the boundary
              gets cut in half. Overlap (50 words) means consecutive chunks
              share some text at the edges, so no sentence is lost.

These chunks will later be turned into "embeddings" (numerical vectors) and
stored in a vector database for semantic search. More on that in embedding.py!
"""

import re
import os
import pdfplumber


# ─────────────────────────────────────────────
# 1. EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and return all its text as a single string.

    How it works:
    - pdfplumber reads each page and uses layout analysis to guess
      where the text characters are and in what order.
    - Some pages may be scanned images (no selectable text), in which
      case page.extract_text() returns None. We skip those safely.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A single string with all extracted text concatenated
        (pages separated by newlines).
    """
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()

            # Safety: skip pages that are None (scanned images)
            # or empty strings (blank pages)
            if page_text and page_text.strip():
                all_text.append(page_text)
            # else: page is empty or image-only — we silently skip it

    return "\n".join(all_text)


# ─────────────────────────────────────────────
# 2. CLEANING
# ─────────────────────────────────────────────

def clean_text(raw_text: str) -> str:
    """
    Normalize messy PDF text into clean, readable prose.

    What we fix:
    - Multiple newlines   → single space  (PDFs break lines per visual row)
    - Multiple spaces     → single space  (column-aligned text leaves gaps)
    - Leading/trailing ws → stripped

    🧠 NLP NOTE:
    Cleaning is critical because embedding models (like sentence-transformers)
    are sensitive to noise. Extra whitespace can change token counts and reduce
    the quality of semantic similarity scores.

    Args:
        raw_text: The raw text extracted from the PDF.

    Returns:
        A single cleaned string.
    """
    if not raw_text:
        return ""

    # Replace newlines with spaces (PDF lines ≠ sentence boundaries)
    text = raw_text.replace("\n", " ")

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ─────────────────────────────────────────────
# 3. CHUNKING
# ─────────────────────────────────────────────

def split_into_chunks(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    🧠 NLP NOTE — Why 400 words?
    ─────────────────────────────
    Most embedding models (e.g. all-MiniLM-L6-v2) work best with input
    that is ≤ 256–512 tokens. 400 *words* ≈ 500–600 tokens, which is a
    good sweet spot: large enough to carry meaning, small enough to
    embed accurately.

    How the sliding window works (example with chunk_size=10, overlap=3):
    ┌──────────────────┐
    │ words 0–9         │  ← chunk 1
    └──────────────────┘
           ┌──────────────────┐
           │ words 7–16        │  ← chunk 2  (starts 7 = 10-3)
           └──────────────────┘
                  ┌──────────────────┐
                  │ words 14–23       │  ← chunk 3
                  └──────────────────┘

    The 3-word overlap means the boundary words appear in BOTH chunks,
    so no context is lost at the seams.

    Args:
        text:       The cleaned text to chunk.
        chunk_size: Number of words per chunk (default: 400).
        overlap:    Number of words shared between consecutive chunks
                    (default: 50).

    Returns:
        A list of text-chunk strings.
    """
    if not text or not text.strip():
        return []

    words = text.split()

    # If the text is shorter than one chunk, return it as-is
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    step = chunk_size - overlap  # how far we advance each iteration

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # If this chunk reached the end of the text, stop
        if end >= len(words):
            break

    return chunks


# ─────────────────────────────────────────────
# 4. CONVENIENCE WRAPPER
# ─────────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[str]:
    """
    Full pipeline: PDF → raw text → cleaned text → chunks.

    This is the single entry-point you'll call from other modules.

    Args:
        pdf_path:   Path to the PDF file.
        chunk_size: Words per chunk.
        overlap:    Word overlap between consecutive chunks.

    Returns:
        List of text chunks ready for embedding.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw_text)
    chunks = split_into_chunks(cleaned, chunk_size, overlap)
    return chunks


def process_directory(
    data_dir: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[dict]:
    """
    Scan a directory for all PDFs and process each one.
    Returns a list of dicts with 'text' and 'source' keys.

    🧠 NLP NOTE — Source Tracking:
    ──────────────────────────────
    When the bot answers a question, users want to know WHERE the
    information came from. By tracking the source PDF name alongside
    each chunk, we can cite our sources — just like a research paper.
    """
    all_chunks = []

    if not os.path.isdir(data_dir):
        print(f"⚠️ Warning: {data_dir} is not a valid directory.")
        return []

    pdf_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    print(f"🔍 Found {len(pdf_files)} PDF files in {data_dir}")

    for filename in pdf_files:
        path = os.path.join(data_dir, filename)
        print(f"📄 Processing {filename}...")
        chunks = process_pdf(path, chunk_size, overlap)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": filename,
            })

    return all_chunks



# ─────────────────────────────────────────────
# Quick demo (runs only when executed directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.utils <path-to-pdf-or-directory>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        chunks = process_directory(path)
    else:
        chunks = process_pdf(path)

    print(f"\n📦 Success! Total chunks created: {len(chunks)}")
    if chunks:
        print(f"\n── First chunk (preview) ──")
        print(chunks[0][:300] + "..." if len(chunks[0]) > 300 else chunks[0])
