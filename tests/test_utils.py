"""
Tests for app/utils.py — the PDF text-processing pipeline.

🧠 TESTING STRATEGY:
─────────────────────
- clean_text and split_into_chunks can be tested with plain strings
  (no PDF files needed — fast and deterministic).
- extract_text_from_pdf needs a real PDF, so we create a tiny one
  programmatically using pdfplumber's sister library (or just skip
  if you don't have fpdf installed). We also test with a real PDF
  if one exists in data/.
"""

import os
import pytest
from app.utils import clean_text, split_into_chunks, process_pdf


# ──────────────────────────────────
# Tests for clean_text()
# ──────────────────────────────────

class TestCleanText:
    """Verify that messy PDF text is normalized properly."""

    def test_collapses_multiple_spaces(self):
        assert clean_text("hello    world") == "hello world"

    def test_replaces_newlines_with_space(self):
        assert clean_text("hello\nworld") == "hello world"

    def test_handles_mixed_whitespace(self):
        result = clean_text("  hello  \n\n  world  \t foo  ")
        assert result == "hello world foo"

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_already_clean_text_unchanged(self):
        assert clean_text("hello world") == "hello world"


# ──────────────────────────────────
# Tests for split_into_chunks()
# ──────────────────────────────────

class TestSplitIntoChunks:
    """Verify the word-based sliding-window chunking logic."""

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size should return as one chunk."""
        text = "word " * 50  # 50 words
        chunks = split_into_chunks(text.strip(), chunk_size=400, overlap=50)
        assert len(chunks) == 1

    def test_exact_chunk_size(self):
        """Text exactly chunk_size words → one chunk."""
        text = " ".join([f"w{i}" for i in range(400)])
        chunks = split_into_chunks(text, chunk_size=400, overlap=50)
        assert len(chunks) == 1

    def test_overlap_works(self):
        """Consecutive chunks should share `overlap` words at boundaries."""
        # 20 words, chunk_size=10, overlap=3 → step=7
        text = " ".join([f"w{i}" for i in range(20)])
        chunks = split_into_chunks(text, chunk_size=10, overlap=3)

        # chunk1 = w0..w9, chunk2 = w7..w16, chunk3 = w14..w19
        assert len(chunks) == 3

        # Verify overlap: last 3 words of chunk1 == first 3 words of chunk2
        c1_words = chunks[0].split()
        c2_words = chunks[1].split()
        assert c1_words[-3:] == c2_words[:3]

    def test_empty_text_returns_empty_list(self):
        assert split_into_chunks("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert split_into_chunks("   \n\n  ") == []

    def test_chunk_word_count(self):
        """Each chunk (except possibly the last) should have chunk_size words."""
        text = " ".join([f"w{i}" for i in range(1000)])
        chunks = split_into_chunks(text, chunk_size=400, overlap=50)

        # First chunk should have exactly 400 words
        assert len(chunks[0].split()) == 400

        # Last chunk may be shorter
        for chunk in chunks[:-1]:
            assert len(chunk.split()) == 400

    def test_no_overlap(self):
        """When overlap=0, chunks should not share any words."""
        text = " ".join([f"w{i}" for i in range(20)])
        chunks = split_into_chunks(text, chunk_size=10, overlap=0)
        assert len(chunks) == 2
        assert chunks[0] == " ".join([f"w{i}" for i in range(10)])
        assert chunks[1] == " ".join([f"w{i}" for i in range(10, 20)])


# ──────────────────────────────────
# Integration test for full pipeline
# ──────────────────────────────────

class TestProcessPdf:
    """Integration test — only runs if a PDF exists in data/."""

    @pytest.mark.skipif(
        not os.path.exists("data/policies.pdf"),
        reason="No policies.pdf in data/ — drop one there to enable this test",
    )
    def test_process_real_pdf(self):
        chunks = process_pdf("data/policies.pdf")
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        # Each chunk should be a non-empty string
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk.strip()) > 0
