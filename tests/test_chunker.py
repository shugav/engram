"""Tests for engram.chunker -- text chunking, hashing, and deduplication."""

from __future__ import annotations

from engram.chunker import chunk_hash, chunk_text, is_duplicate, jaccard_similarity


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_long_text_splits_into_multiple(self):
        sentences = [f"Sentence number {i} is here." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) > 1

    def test_overlap_preserves_context(self):
        sentences = [f"Important fact {i} for context." for i in range(30)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=80, overlap_tokens=20)

        if len(chunks) >= 2:
            words_first = set(chunks[0].split())
            words_second = set(chunks[1].split())
            overlap = words_first & words_second
            assert len(overlap) > 0, "Chunks should have overlapping words"

    def test_single_long_sentence(self):
        text = "word " * 1000
        chunks = chunk_text(text.strip(), max_tokens=100)
        assert len(chunks) >= 1


class TestChunkHash:
    def test_deterministic(self):
        h1 = chunk_hash("Hello world")
        h2 = chunk_hash("Hello world")
        assert h1 == h2

    def test_whitespace_normalized(self):
        h1 = chunk_hash("hello  world")
        h2 = chunk_hash("hello world")
        assert h1 == h2

    def test_case_normalized(self):
        h1 = chunk_hash("Hello World")
        h2 = chunk_hash("hello world")
        assert h1 == h2

    def test_different_text_different_hash(self):
        h1 = chunk_hash("alpha")
        h2 = chunk_hash("beta")
        assert h1 != h2


class TestJaccardSimilarity:
    def test_identical_strings(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert jaccard_similarity("alpha beta", "gamma delta") == 0.0

    def test_partial_overlap(self):
        sim = jaccard_similarity("the quick brown fox", "the slow brown dog")
        assert 0.0 < sim < 1.0

    def test_empty_string(self):
        assert jaccard_similarity("", "hello") == 0.0
        assert jaccard_similarity("", "") == 0.0


class TestIsDuplicate:
    def test_exact_duplicate(self):
        existing = ["Hello world this is a test"]
        assert is_duplicate("Hello world this is a test", existing) is True

    def test_near_duplicate(self):
        # 13/15 unique words overlap -> Jaccard ~0.867, above the 0.85 threshold
        existing = ["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"]
        candidate = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu omicron"
        assert is_duplicate(candidate, existing) is True

    def test_not_duplicate(self):
        existing = ["Quantum computing fundamentals"]
        assert is_duplicate("Web development with Python", existing) is False

    def test_empty_existing(self):
        assert is_duplicate("Anything", []) is False
