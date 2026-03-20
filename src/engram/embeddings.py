"""Embedding providers for engram.

Supports three backends:
  - OpenAI (premium, highest quality): text-embedding-3-small, 1536 dims
  - Ollama (local, free, good quality): nomic-embed-text via REST API, 768 dims
  - Null (no vector search): BM25-only mode, zero dependencies

Selected via ENGRAM_EMBEDDER env var (openai|ollama|none). Default: auto-detect.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import struct
from typing import Protocol, Sequence, runtime_checkable
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)


def validate_ollama_base_url(url: str) -> str:
    """Validate Ollama base URL to reduce SSRF risk (issue #19).

    Allows http/https only. Blocks link-local / cloud metadata targets often
    abused in SSRF. Resolves hostnames once and rejects 169.254.0.0/16.

    Raises:
        ValueError: If the URL is not allowed.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("OLLAMA_URL is empty")
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"OLLAMA_URL scheme must be http or https, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("OLLAMA_URL must include a hostname")
    h = host.lower()
    if h in ("metadata.google.internal", "metadata.google.com"):
        raise ValueError("OLLAMA_URL host is not allowed")
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"OLLAMA_URL host does not resolve: {e}") from e
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if ip in ipaddress.ip_network("169.254.0.0/16"):
            raise ValueError(
                "OLLAMA_URL resolves to link-local/metadata range 169.254.0.0/16; not allowed"
            )
    return raw.rstrip("/")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol that all embedding backends must implement (async v3)."""

    name: str
    dimensions: int

    async def embed(self, text: str) -> np.ndarray: ...

    async def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]: ...


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small (1536 dimensions)."""

    name = "openai/text-embedding-3-small"
    dimensions = 1536
    version = "v1"

    def __init__(self, api_key: str | None = None):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)

    async def embed(self, text: str) -> np.ndarray:
        resp = await self._client.embeddings.create(input=[text], model="text-embedding-3-small")
        return np.array(resp.data[0].embedding, dtype=np.float32)

    async def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = await self._client.embeddings.create(input=batch, model="text-embedding-3-small")
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            all_embeddings.extend(np.array(d.embedding, dtype=np.float32) for d in sorted_data)
        return all_embeddings


class OllamaEmbedder:
    """Ollama embedding via local REST API.

    Calls Ollama's /api/embed endpoint directly with httpx -- no ollama
    Python package needed.  Model and dimensions are configurable; defaults
    to nomic-embed-text (768 dims) for backward compatibility.
    """

    version = "v1.5"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimensions: int = 768,
    ):
        self._base_url = validate_ollama_base_url(base_url).rstrip("/")
        self._model = model
        self.name = f"ollama/{model}"
        self.dimensions = dimensions

    async def embed(self, text: str) -> np.ndarray:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            return np.array(data["embeddings"][0], dtype=np.float32)

    async def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        import httpx

        all_embeddings: list[np.ndarray] = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i : i + batch_size])
                resp = await client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                all_embeddings.extend(np.array(emb, dtype=np.float32) for emb in data["embeddings"])
        return all_embeddings


class NullEmbedder:
    """No-op embedder for BM25-only mode. Zero external dependencies."""

    name = "none"
    dimensions = 0
    version = "n/a"

    async def embed(self, text: str) -> np.ndarray:
        return np.array([], dtype=np.float32)

    async def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        return [np.array([], dtype=np.float32) for _ in texts]


def create_embedder(
    provider: str | None = None,
    api_key: str | None = None,
    ollama_url: str = "http://localhost:11434",
) -> EmbeddingProvider:
    """Factory that creates the appropriate embedder based on config.

    Args:
        provider: "openai", "ollama", "none", or None for auto-detect.
        api_key: OpenAI API key (only needed for openai provider).
        ollama_url: Ollama base URL (only needed for ollama provider).

    Auto-detect order: Ollama (if reachable) -> OpenAI (if key set) -> None.
    """
    if provider is None:
        provider = os.environ.get("ENGRAM_EMBEDDER", "").strip().lower()

    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            logger.warning("OPENAI_API_KEY not set, falling back to BM25-only mode")
            return NullEmbedder()
        return OpenAIEmbedder(api_key=key)

    if provider == "ollama":
        url = os.environ.get("OLLAMA_URL", ollama_url)
        return OllamaEmbedder(base_url=validate_ollama_base_url(url))

    if provider == "none":
        return NullEmbedder()

    # Auto-detect
    auto_url = os.environ.get("OLLAMA_URL", ollama_url)
    try:
        auto_url = validate_ollama_base_url(auto_url)
    except ValueError as e:
        logger.warning("Invalid OLLAMA_URL for auto-detect: %s", e)
        auto_url = ""
    if auto_url and _ollama_reachable(auto_url):
        logger.info("Auto-detected Ollama at %s, using local embeddings", auto_url)
        return OllamaEmbedder(base_url=auto_url)

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        logger.info("Using OpenAI embeddings")
        return OpenAIEmbedder(api_key=key)

    logger.info("No embedding provider available, using BM25-only mode")
    return NullEmbedder()


def _ollama_reachable(base_url: str) -> bool:
    """Quick check if Ollama is running and has nomic-embed-text."""
    try:
        import httpx
    except ImportError:
        logger.debug("httpx not installed; cannot probe Ollama")
        return False
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any("nomic-embed-text" in m for m in models)
    except httpx.HTTPError:
        pass
    except Exception:
        logger.debug("Ollama reachability check failed", exc_info=True)
    return False


# ── Serialization helpers (unchanged) ────────────────────────────


def to_blob(vec: np.ndarray) -> bytes:
    if len(vec) == 0:
        return b""
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def from_blob(blob: bytes) -> np.ndarray:
    if not blob:
        return np.array([], dtype=np.float32)
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)} (vectors must match)")
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# pgvector column is fixed 1536 dims; shorter provider vectors are zero-padded.
EMBEDDING_DIM = 1536


def format_vector_literal(data: np.ndarray | bytes | memoryview) -> str:
    """Format a float32 vector as a Postgres ``vector`` literal (no extra deps)."""
    if isinstance(data, (bytes, memoryview)):
        arr = from_blob(bytes(data))
    else:
        arr = np.asarray(data, dtype=np.float32).flatten()
    out = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    n = min(len(arr), EMBEDDING_DIM)
    if n > 0:
        out[:n] = arr[:n]
    return "[" + ",".join(f"{float(x):.8g}" for x in out) + "]"


def vector_to_numpy_bytes(value: object) -> bytes:
    """Convert a pgvector / list / string literal to float32 bytes for ``Chunk.embedding``."""
    if value is None:
        return b""
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return b""
        arr = np.array([float(x) for x in s.split(",")], dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32).flatten()
    if arr.size == 0:
        return b""
    return struct.pack(f"{arr.size}f", *arr.tolist())
