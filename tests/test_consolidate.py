"""Stress tests for memory_consolidate (async Postgres)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from engram.chunker import chunk_hash
from engram.db import MemoryDB
from engram.embeddings import to_blob
from engram.search import SearchEngine
from engram.types import Chunk, Memory, Relationship, RelationType
from tests.conftest import FakeEmbedder, set_memory_last_accessed


@pytest_asyncio.fixture
async def stress_engine(pg_pool) -> SearchEngine:
    async with pg_pool.connection() as conn:
        await conn.execute("TRUNCATE memories, project_meta CASCADE")
    db = MemoryDB(project="stress", pool=pg_pool)
    return SearchEngine(db=db, embedder=FakeEmbedder())


@pytest.mark.asyncio
class TestChunkDeduplication:
    async def test_dedup_removes_duplicate_chunks(self, stress_engine: SearchEngine):
        m = await stress_engine.store(Memory(content="Base memory for dedup test"))
        blob = to_blob(await stress_engine.embedder.embed("dummy"))
        the_hash = chunk_hash("This exact chunk appears many times")
        chunks: list[Chunk] = []
        for i in range(20):
            chunks.append(
                Chunk(
                    memory_id=m.id,
                    chunk_text="This exact chunk appears many times",
                    chunk_index=i + 10,
                    chunk_hash=the_hash,
                    embedding=blob,
                )
            )
        await stress_engine.db.store_chunks(chunks)

        result = await stress_engine.memify()
        assert result["chunks_deduped"] >= 19

    async def test_dedup_preserves_unique_chunks(self, stress_engine: SearchEngine):
        for i in range(100):
            await stress_engine.store(
                Memory(content=f"Unique memory number {i} about topic {i * 7}")
            )

        await stress_engine.memify()
        stats_after = await stress_engine.db.get_stats()

        assert stats_after.total_chunks >= 100


@pytest.mark.asyncio
class TestEdgeDecayAndPruning:
    async def test_weak_edges_pruned(self, stress_engine: SearchEngine):
        memories = []
        for i in range(20):
            m = Memory(content=f"Memory {i} for edge testing")
            memories.append(await stress_engine.store(m))

        for i in range(0, 18, 2):
            rel = Relationship(
                source_id=memories[i].id,
                target_id=memories[i + 1].id,
                rel_type=RelationType.RELATES_TO,
                strength=0.05 if i < 10 else 0.8,
            )
            await stress_engine.db.store_relationship(rel)

        result = await stress_engine.memify()

        assert result["edges_decayed"] >= 9
        assert result["edges_pruned"] >= 1

    async def test_strong_edges_survive(self, stress_engine: SearchEngine):
        m1 = await stress_engine.store(Memory(content="Strong edge source"))
        m2 = await stress_engine.store(Memory(content="Strong edge target"))

        rel = Relationship(
            source_id=m1.id,
            target_id=m2.id,
            rel_type=RelationType.DEPENDS_ON,
            strength=1.0,
        )
        await stress_engine.db.store_relationship(rel)

        await stress_engine.memify()

        connected = await stress_engine.db.get_connected(m1.id)
        assert len(connected) == 1
        assert connected[0][3] > 0.1

    async def test_50_edges_decay_correctly(self, stress_engine: SearchEngine):
        memories = []
        for i in range(51):
            memories.append(await stress_engine.store(Memory(content=f"Node {i}")))

        for i in range(50):
            rel = Relationship(
                source_id=memories[i].id,
                target_id=memories[i + 1].id,
                strength=0.5,
            )
            await stress_engine.db.store_relationship(rel)

        result = await stress_engine.memify()
        assert result["edges_decayed"] == 50


@pytest.mark.asyncio
class TestStalePruning:
    async def test_old_unaccessed_trivial_pruned(self, stress_engine: SearchEngine):
        old_date = datetime.now(timezone.utc) - timedelta(days=45)

        for i in range(30):
            m = Memory(content=f"Stale memory {i}", importance=0)
            stored = await stress_engine.db.store_memory(m)
            await set_memory_last_accessed(stress_engine.db, stored.id, old_date)

        result = await stress_engine.memify()
        assert result["stale_memories_pruned"] >= 25

    async def test_important_old_memories_survive(self, stress_engine: SearchEngine):
        old_date = datetime.now(timezone.utc) - timedelta(days=90)

        for i in range(10):
            m = Memory(content=f"Critical decision {i}", importance=4)
            stored = await stress_engine.db.store_memory(m)
            await set_memory_last_accessed(stress_engine.db, stored.id, old_date)

        result = await stress_engine.memify()
        assert result["stale_memories_pruned"] == 0

        stats = await stress_engine.db.get_stats()
        assert stats.total_memories == 10

    async def test_accessed_memories_survive(self, stress_engine: SearchEngine):
        old_date = datetime.now(timezone.utc) - timedelta(days=45)

        for i in range(10):
            m = Memory(content=f"Accessed memory {i}", importance=0)
            stored = await stress_engine.db.store_memory(m)
            await stress_engine.db.touch_memory(stored.id)
            await set_memory_last_accessed(stress_engine.db, stored.id, old_date)

        result = await stress_engine.memify()
        assert result["stale_memories_pruned"] == 0


@pytest.mark.asyncio
class TestIdempotency:
    async def test_double_consolidation(self, stress_engine: SearchEngine):
        for i in range(50):
            await stress_engine.store(Memory(content=f"Memory {i} for idempotency test"))

        await stress_engine.memify()
        result2 = await stress_engine.memify()

        assert result2["chunks_deduped"] == 0
        assert result2["edges_pruned"] == 0
        assert result2["stale_memories_pruned"] == 0


@pytest.mark.asyncio
class TestPerformanceBenchmark:
    @pytest.mark.slow
    async def test_consolidation_timing(self, stress_engine: SearchEngine):
        old_date = datetime.now(timezone.utc) - timedelta(days=45)

        for i in range(200):
            m = Memory(
                content=f"Benchmark memory {i} about various topics like auth databases and APIs",
                importance=3 if i % 3 == 0 else 2,
            )
            stored = await stress_engine.db.store_memory(m)
            if i % 3 == 0:
                await set_memory_last_accessed(stress_engine.db, stored.id, old_date)

        memories = await stress_engine.db.list_memories(limit=200)
        for i in range(0, min(len(memories) - 1, 60), 2):
            rel = Relationship(
                source_id=memories[i].id,
                target_id=memories[i + 1].id,
                strength=0.3 if i < 30 else 0.8,
            )
            await stress_engine.db.store_relationship(rel)

        start = time.perf_counter()
        result = await stress_engine.memify()
        elapsed = time.perf_counter() - start

        print("\n=== Consolidation Benchmark ===")
        print("  Memories: 200")
        print("  Edges: ~30")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Chunks deduped: {result['chunks_deduped']}")
        print(f"  Edges decayed: {result['edges_decayed']}")
        print(f"  Edges pruned: {result['edges_pruned']}")
        print(f"  Stale pruned: {result['stale_memories_pruned']}")
