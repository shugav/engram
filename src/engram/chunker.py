from __future__ import annotations

import hashlib
import re


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Uses a rough 1 token ~ 4 chars approximation to avoid a tokenizer dependency.
    """
    if not text.strip():
        return []

    chars_per_token = 4
    max_chars = max_tokens * chars_per_token
    overlap_chars = overlap_tokens * chars_per_token

    sentences = _split_sentences(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        slen = len(sentence)

        if current_len + slen > max_chars and current:
            chunks.append(" ".join(current))

            # Build overlap from the tail of the current chunk
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap_chars:
                    break
                overlap.insert(0, s)
                overlap_len += len(s)
            current = overlap
            current_len = overlap_len

        current.append(sentence)
        current_len += slen

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def chunk_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(re.sub(r"\s+", " ", a.strip().lower()).split())
    words_b = set(re.sub(r"\s+", " ", b.strip().lower()).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def is_duplicate(new_text: str, existing_texts: list[str], threshold: float = 0.85) -> bool:
    for existing in existing_texts:
        if jaccard_similarity(new_text, existing) >= threshold:
            return True
    return False


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, keeping the delimiter attached."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
