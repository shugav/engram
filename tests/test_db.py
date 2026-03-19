"""Tests for engram.db.MemoryDB (Postgres async)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from engram.db import MemoryDB
from engram.types import Memory, MemoryType, Relationship, RelationType
from tests.conftest import set_memory_last_accessed


@pytest.mark.asyncio
class TestMemoryCRUD:
    async def test_store_and_retrieve(self, db: MemoryDB):
        mem = Memory(content="PostgreSQL chosen for the main database")
        stored = await db.store_memory(mem)

        retrieved = await db.get_memory(stored.id)
        assert retrieved is not None
        assert retrieved.content == "PostgreSQL chosen for the main database"
        assert retrieved.project == "test"

    async def test_get_nonexistent_returns_none(self, db: MemoryDB):
        assert await db.get_memory("does-not-exist") is None

    async def test_update_memory(self, db: MemoryDB):
        mem = Memory(content="Old content", tags=["old"])
        stored = await db.store_memory(mem)

        updated = await db.update_memory(stored.id, content="New content", tags=["new"])
        assert updated is not None
        assert updated.content == "New content"
        assert updated.tags == ["new"]

    async def test_delete_memory(self, db: MemoryDB):
        mem = Memory(content="To be deleted")
        stored = await db.store_memory(mem)

        assert await db.delete_memory(stored.id) is True
        assert await db.get_memory(stored.id) is None

    async def test_delete_nonexistent_returns_false(self, db: MemoryDB):
        assert await db.delete_memory("nope") is False

    async def test_touch_increments_access(self, db: MemoryDB):
        mem = Memory(content="Touch me")
        stored = await db.store_memory(mem)

        await db.touch_memory(stored.id)
        await db.touch_memory(stored.id)
        retrieved = await db.get_memory(stored.id)
        assert retrieved is not None
        assert retrieved.access_count == 2

    async def test_list_memories_filters_by_type(self, db: MemoryDB):
        await db.store_memory(Memory(content="A decision", memory_type=MemoryType.DECISION))
        await db.store_memory(Memory(content="An error", memory_type=MemoryType.ERROR))
        await db.store_memory(Memory(content="A pattern", memory_type=MemoryType.PATTERN))

        decisions = await db.list_memories(memory_type=MemoryType.DECISION)
        assert len(decisions) == 1
        assert decisions[0].memory_type == MemoryType.DECISION

    async def test_list_memories_filters_by_importance(self, db: MemoryDB):
        await db.store_memory(Memory(content="Critical", importance=4))
        await db.store_memory(Memory(content="Trivial", importance=0))

        high = await db.list_memories(min_importance=4)
        assert len(high) == 1
        assert high[0].content == "Critical"

    async def test_list_memories_filters_by_tags(self, db: MemoryDB):
        await db.store_memory(Memory(content="Auth stuff", tags=["auth", "jwt"]))
        await db.store_memory(Memory(content="DB stuff", tags=["postgres", "sql"]))

        auth = await db.list_memories(tags=["auth"])
        assert len(auth) == 1
        assert "auth" in auth[0].tags


@pytest.mark.asyncio
class TestProjectIsolation:
    async def test_same_pool_different_projects(self, pg_pool):
        async with pg_pool.connection() as conn:
            await conn.execute("TRUNCATE memories, project_meta CASCADE")

        db_a = MemoryDB(project="alpha", pool=pg_pool)
        db_b = MemoryDB(project="beta", pool=pg_pool)

        await db_a.store_memory(Memory(content="Alpha secret"))
        await db_b.store_memory(Memory(content="Beta secret"))

        alpha_mems = await db_a.list_memories()
        beta_mems = await db_b.list_memories()

        assert len(alpha_mems) == 1
        assert alpha_mems[0].content == "Alpha secret"
        assert len(beta_mems) == 1
        assert beta_mems[0].content == "Beta secret"

    async def test_fts_isolated_between_projects(self, pg_pool):
        async with pg_pool.connection() as conn:
            await conn.execute("TRUNCATE memories, project_meta CASCADE")

        db_a = MemoryDB(project="alpha", pool=pg_pool)
        db_b = MemoryDB(project="beta", pool=pg_pool)

        await db_a.store_memory(Memory(content="Alpha uses PostgreSQL for everything"))

        results = await db_b.fts_search("PostgreSQL")
        assert len(results) == 0


@pytest.mark.asyncio
class TestFTSSearch:
    async def test_basic_search(self, db: MemoryDB):
        await db.store_memory(Memory(content="JWT authentication with refresh tokens"))
        await db.store_memory(Memory(content="Database migration using alembic"))

        results = await db.fts_search("JWT authentication")
        assert len(results) >= 1
        assert "JWT" in results[0][0].content

    async def test_empty_query_returns_empty(self, db: MemoryDB):
        await db.store_memory(Memory(content="Some content"))
        results = await db.fts_search("")
        assert results == []

    async def test_no_match_returns_empty(self, db: MemoryDB):
        await db.store_memory(Memory(content="Python web framework"))
        results = await db.fts_search("quantum entanglement")
        assert len(results) == 0


@pytest.mark.asyncio
class TestRelationships:
    async def test_store_and_get_connected(self, db: MemoryDB):
        m1 = await db.store_memory(Memory(content="Memory A"))
        m2 = await db.store_memory(Memory(content="Memory B"))

        rel = Relationship(
            source_id=m1.id,
            target_id=m2.id,
            rel_type=RelationType.RELATES_TO,
            strength=0.8,
        )
        await db.store_relationship(rel)

        connected = await db.get_connected(m1.id, max_hops=1)
        assert len(connected) == 1
        assert connected[0][0].id == m2.id
        assert connected[0][1] == "relates_to"

    async def test_supersedes_relationship(self, db: MemoryDB):
        old = await db.store_memory(Memory(content="Use MySQL"))
        new = await db.store_memory(Memory(content="Use PostgreSQL instead"))

        rel = Relationship(
            source_id=new.id,
            target_id=old.id,
            rel_type=RelationType.SUPERSEDES,
        )
        await db.store_relationship(rel)

        connected = await db.get_connected(old.id, max_hops=1)
        assert len(connected) == 1
        assert connected[0][1] == "supersedes"

    async def test_boost_and_decay_edges(self, db: MemoryDB):
        m1 = await db.store_memory(Memory(content="A"))
        m2 = await db.store_memory(Memory(content="B"))

        rel = Relationship(
            source_id=m1.id,
            target_id=m2.id,
            rel_type=RelationType.RELATES_TO,
            strength=0.5,
        )
        await db.store_relationship(rel)

        await db.boost_edges_for_memory(m1.id, factor=0.2)
        connected = await db.get_connected(m1.id)
        assert connected[0][3] == pytest.approx(0.7, abs=0.01)

        await db.decay_edges_for_memory(m1.id, factor=0.3)
        connected = await db.get_connected(m1.id)
        assert connected[0][3] == pytest.approx(0.4, abs=0.01)

    async def test_delete_relationships_for_memory(self, db: MemoryDB):
        m1 = await db.store_memory(Memory(content="A"))
        m2 = await db.store_memory(Memory(content="B"))

        rel = Relationship(source_id=m1.id, target_id=m2.id)
        await db.store_relationship(rel)

        await db.delete_relationships_for_memory(m1.id)
        assert await db.get_connection_count(m1.id) == 0


@pytest.mark.asyncio
class TestStats:
    async def test_stats_reflect_stored_data(self, db: MemoryDB):
        await db.store_memory(Memory(content="Decision 1", memory_type=MemoryType.DECISION))
        await db.store_memory(Memory(content="Error 1", memory_type=MemoryType.ERROR))

        stats = await db.get_stats()
        assert stats.total_memories == 2
        assert stats.by_type.get("decision") == 1
        assert stats.by_type.get("error") == 1


@pytest.mark.asyncio
class TestPruning:
    async def test_prune_stale_memories(self, db: MemoryDB):
        old = Memory(content="Old and forgotten", importance=0)
        stored = await db.store_memory(old)

        cutoff = datetime.now(timezone.utc) - timedelta(days=31)
        await set_memory_last_accessed(db, stored.id, cutoff)

        pruned = await db.prune_stale_memories(max_age_hours=720, max_importance=1)
        assert pruned == 1
        assert await db.get_memory(stored.id) is None

    async def test_important_memories_survive_pruning(self, db: MemoryDB):
        important = Memory(content="Critical decision", importance=4)
        stored = await db.store_memory(important)

        cutoff = datetime.now(timezone.utc) - timedelta(days=60)
        await set_memory_last_accessed(db, stored.id, cutoff)

        pruned = await db.prune_stale_memories(max_age_hours=720, max_importance=1)
        assert pruned == 0
        assert await db.get_memory(stored.id) is not None
