"""Shared fixtures: Postgres pool (v3), FakeEmbedder, SearchEngine."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pytest
import pytest_asyncio

from engram.db import MemoryDB
from engram.embeddings import EMBEDDING_DIM
from engram.search import SearchEngine


class FakeEmbedder:
    """Deterministic async embedder (no API calls). Vectors padded to ``EMBEDDING_DIM``."""

    name = "fake/test-embedder"
    dimensions = EMBEDDING_DIM
    version = "v1-test"

    async def embed(self, text: str) -> np.ndarray:
        words = set(text.lower().split())
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for w in words:
            idx = hash(w) % EMBEDDING_DIM
            vec[idx] = 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    async def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        return [await self.embed(t) for t in texts]


def _require_test_dsn() -> str:
    dsn = os.environ.get("ENGRAM_TEST_DATABASE_URL", "").strip()
    if not dsn:
        pytest.skip(
            "Set ENGRAM_TEST_DATABASE_URL to a Postgres URL with pgvector "
            "(e.g. postgresql://engram:engram@127.0.0.1:5432/engram_test)"
        )
    return dsn


@pytest_asyncio.fixture(scope="session")
async def pg_pool():
    """One async pool per test session; applies migrations once."""
    dsn = _require_test_dsn()
    os.environ["DATABASE_URL"] = dsn

    import psycopg

    with psycopg.connect(dsn, autocommit=True) as reset_conn:
        with reset_conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
            cur.execute("GRANT ALL ON SCHEMA public TO PUBLIC")

    import engram.pool as pool_mod

    pool_mod._pool = None
    from engram.pool import close_pool, get_pool

    pool = await get_pool()
    yield pool
    await close_pool()
    pool_mod._pool = None


@pytest_asyncio.fixture
async def db(pg_pool):
    """Empty ``test`` project namespace before each test."""
    async with pg_pool.connection() as conn:
        await conn.execute("TRUNCATE memories, project_meta CASCADE")
    return MemoryDB("test", pg_pool)


@pytest_asyncio.fixture
async def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest_asyncio.fixture
async def engine(db: MemoryDB, embedder: FakeEmbedder) -> SearchEngine:
    return SearchEngine(db=db, embedder=embedder)


@pytest_asyncio.fixture
async def tmp_db_dir():  # kept for API compat; unused under Postgres
    return None


async def set_memory_last_accessed(db: MemoryDB, memory_id: str, when) -> None:
    """Test helper: backdate ``last_accessed``."""
    async with db.pool.connection() as conn:
        await conn.execute(
            "UPDATE memories SET last_accessed = %s WHERE id = %s AND project = %s",
            (when, memory_id, db.project),
        )
