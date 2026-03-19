"""Tests for engram.embeddings (async providers)."""

from __future__ import annotations

import numpy as np
import pytest

from engram.embeddings import (
    EMBEDDING_DIM,
    NullEmbedder,
    cosine_similarity,
    create_embedder,
    from_blob,
    to_blob,
    validate_ollama_base_url,
)
from engram.errors import EmbeddingConfigMismatchError
from engram.search import SearchEngine
from engram.types import Memory
from tests.conftest import FakeEmbedder


class TestNullEmbedder:
    @pytest.mark.asyncio
    async def test_embed_returns_empty(self):
        emb = NullEmbedder()
        vec = await emb.embed("any text")
        assert len(vec) == 0

    @pytest.mark.asyncio
    async def test_embed_batch_returns_empty_list(self):
        emb = NullEmbedder()
        results = await emb.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(v) == 0 for v in results)

    def test_name_and_dimensions(self):
        emb = NullEmbedder()
        assert emb.name == "none"
        assert emb.dimensions == 0


class TestFakeEmbedder:
    @pytest.mark.asyncio
    async def test_deterministic(self):
        emb = FakeEmbedder()
        v1 = await emb.embed("hello world")
        v2 = await emb.embed("hello world")
        assert np.array_equal(v1, v2)

    @pytest.mark.asyncio
    async def test_similar_texts_high_similarity(self):
        emb = FakeEmbedder()
        v1 = await emb.embed("database PostgreSQL performance tuning optimization")
        v2 = await emb.embed("database PostgreSQL query optimization performance")
        sim = cosine_similarity(v1, v2)
        assert sim >= 0.5

    @pytest.mark.asyncio
    async def test_different_texts_lower_similarity(self):
        emb = FakeEmbedder()
        v1 = await emb.embed("database PostgreSQL performance")
        v2 = await emb.embed("frontend React TypeScript components")
        sim = cosine_similarity(v1, v2)
        assert sim < 0.5

    def test_has_protocol_fields(self):
        emb = FakeEmbedder()
        assert hasattr(emb, "name")
        assert hasattr(emb, "dimensions")
        assert emb.dimensions == EMBEDDING_DIM


class TestCreateEmbedder:
    def test_none_provider(self):
        emb = create_embedder(provider="none")
        assert isinstance(emb, NullEmbedder)

    def test_openai_without_key_falls_back(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        emb = create_embedder(provider="openai", api_key=None)
        assert isinstance(emb, NullEmbedder)

    def test_env_var_selects_none(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_EMBEDDER", "none")
        emb = create_embedder()
        assert isinstance(emb, NullEmbedder)


class TestBlobSerialization:
    def test_round_trip(self):
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        blob = to_blob(vec)
        restored = from_blob(blob)
        assert np.allclose(vec, restored)

    def test_empty_round_trip(self):
        vec = np.array([], dtype=np.float32)
        blob = to_blob(vec)
        restored = from_blob(blob)
        assert len(restored) == 0

    def test_cosine_empty_returns_zero(self):
        empty = np.array([], dtype=np.float32)
        full = np.array([1.0, 2.0], dtype=np.float32)
        assert cosine_similarity(empty, full) == 0.0
        assert cosine_similarity(full, empty) == 0.0

    def test_cosine_dimension_mismatch_raises(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity(a, b)


@pytest.mark.asyncio
class TestMetadataEnforcement:
    async def test_first_store_sets_metadata(self, db):
        emb = FakeEmbedder()
        engine = SearchEngine(db=db, embedder=emb)

        await engine.store(Memory(content="First memory"))

        assert await db.get_meta("embedder_name") == "fake/test-embedder"
        assert await db.get_meta("embedder_dimensions") == str(EMBEDDING_DIM)

    async def test_same_embedder_succeeds(self, db):
        emb = FakeEmbedder()
        engine = SearchEngine(db=db, embedder=emb)

        await engine.store(Memory(content="First"))
        await engine.store(Memory(content="Second"))

    async def test_different_embedder_raises_error(self, db):
        emb1 = FakeEmbedder()
        engine1 = SearchEngine(db=db, embedder=emb1)
        await engine1.store(Memory(content="Stored with fake embedder"))

        class DifferentEmbedder:
            name = "other/model"
            dimensions = 128

            async def embed(self, text: str) -> np.ndarray:
                return np.zeros(128, dtype=np.float32)

            async def embed_batch(self, texts, batch_size=64):
                return [await self.embed(t) for t in texts]

        engine2 = SearchEngine(db=db, embedder=DifferentEmbedder())

        with pytest.raises(EmbeddingConfigMismatchError) as exc_info:
            await engine2.store(Memory(content="This should fail"))

        assert "fake/test-embedder" in str(exc_info.value)
        assert "other/model" in str(exc_info.value)

    async def test_null_embedder_skips_metadata(self, db):
        emb = NullEmbedder()
        engine = SearchEngine(db=db, embedder=emb)

        await engine.store(Memory(content="BM25 only mode"))

        assert await db.get_meta("embedder_name") is None
        assert await db.get_meta("embedder_dimensions") is None


@pytest.mark.asyncio
class TestNullEmbedderSearch:
    async def test_store_and_recall_bm25_only(self, db):
        emb = NullEmbedder()
        engine = SearchEngine(db=db, embedder=emb)

        await engine.store(Memory(content="PostgreSQL is our main database"))
        results = await engine.recall("PostgreSQL database")

        assert len(results) >= 1
        assert "PostgreSQL" in results[0].memory.content

    async def test_no_vector_score_in_bm25_mode(self, db):
        emb = NullEmbedder()
        engine = SearchEngine(db=db, embedder=emb)

        await engine.store(Memory(content="Authentication uses JWT tokens"))
        results = await engine.recall("JWT authentication")

        if results:
            assert results[0].score_breakdown["vector"] == 0.0

    async def test_bm25_dedup_works_without_embeddings(self, db):
        emb = NullEmbedder()
        engine = SearchEngine(db=db, embedder=emb)

        await engine.store(Memory(content="Exact same content stored twice"))
        await engine.store(Memory(content="Exact same content stored twice"))

        stats = await db.get_stats()
        assert stats.total_chunks == 1


class TestOllamaUrlValidation:
    def test_blocks_link_local_metadata_range(self):
        with pytest.raises(ValueError, match="169.254"):
            validate_ollama_base_url("http://169.254.169.254:11434")

    def test_allows_localhost(self):
        u = validate_ollama_base_url("http://127.0.0.1:11434")
        assert "127.0.0.1" in u


class TestAutoDetectOllamaUrl:
    def test_auto_detect_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_URL", "http://custom-host:11434")
        monkeypatch.delenv("ENGRAM_EMBEDDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        emb = create_embedder()
        assert isinstance(emb, NullEmbedder)

    def test_auto_detect_skips_invalid_ollama_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_URL", "http://169.254.169.254:11434")
        monkeypatch.delenv("ENGRAM_EMBEDDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        emb = create_embedder()
        assert isinstance(emb, NullEmbedder)


@pytest.mark.asyncio
class TestMetadataVersion:
    async def test_version_stored_on_first_embed(self, db):
        emb = FakeEmbedder()
        engine = SearchEngine(db=db, embedder=emb)

        await engine.store(Memory(content="Test version storage"))

        assert await db.get_meta("embedder_version") == "v1-test"
