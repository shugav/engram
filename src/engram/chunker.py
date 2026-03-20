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

        # Hard break: always emit before exceeding max_chars
        hard_break = current_len + slen > max_chars and current
        # Soft break: prefer breaking earlier near paragraph boundaries
        soft_break = (current_len + slen > max_chars * 0.8 and current
                      and len(current) > 1 and current_len > max_chars * 0.3)
        if hard_break or soft_break:
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
    """Split text into sentences, respecting paragraph boundaries first.

    This produces cleaner chunks for structured documents by treating paragraph
    breaks as natural chunk boundaries. First splits on paragraphs (\n\n+),
    then splits sentences within each paragraph.
    """
    if not text.strip():
        return []

    # Split on paragraph boundaries first (double newlines or more)
    paragraphs = re.split(r"\n\s*\n", text.strip())

    sentences: list[str] = []
    for para in paragraphs:
        if not para.strip():
            continue
        # Split sentences within paragraph
        parts = re.split(r"(?<=[.!?])\s+", para.strip())
        sentences.extend(p.strip() for p in parts if p.strip())

    return sentences
