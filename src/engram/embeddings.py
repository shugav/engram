from __future__ import annotations

import struct
from typing import Sequence

import numpy as np
from openai import OpenAI

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


class EmbeddingClient:
    def __init__(self, api_key: str | None = None):
        self._client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(input=[text], model=MODEL)
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = self._client.embeddings.create(input=batch, model=MODEL)
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            all_embeddings.extend(
                np.array(d.embedding, dtype=np.float32) for d in sorted_data
            )
        return all_embeddings


def to_blob(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def from_blob(blob: bytes) -> np.ndarray:
    n = len(blob) // 4  # float32 = 4 bytes
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)
