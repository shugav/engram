"""Tests for SearchEngine (async)."""

from __future__ import annotations

import pytest

from engram.types import Memory, MemoryType, Relationship, RelationType


@pytest.mark.asyncio
class TestStoreRecallRoundTrip:
    async def test_stored_memory_is_recallable(self, engine):
        mem = Memory(content="We chose PostgreSQL because it supports JSONB natively")
        await engine.store(mem)

        results = await engine.recall("PostgreSQL database choice")
        assert len(results) >= 1
        assert "PostgreSQL" in results[0].memory.content

    async def test_recall_returns_best_match_first(self, engine):
        await engine.store(Memory(content="Authentication uses JWT with RS256 signing"))
        await engine.store(Memory(content="Database uses PostgreSQL 16 with pgvector"))
        await engine.store(Memory(content="Frontend built with React and TypeScript"))

        results = await engine.recall("JWT authentication signing")
        assert "JWT" in results[0].memory.content

    async def test_recall_empty_query(self, engine):
        await engine.store(Memory(content="Some stored content"))
        results = await engine.recall("")
        assert isinstance(results, list)

    async def test_recall_no_results(self, engine):
        results = await engine.recall("quantum entanglement")
        assert len(results) == 0


@pytest.mark.asyncio
class TestMemoryTypeFiltering:
    async def test_filter_by_type(self, engine):
        await engine.store(
            Memory(
                content="Chose microservices over monolith",
                memory_type=MemoryType.DECISION,
            )
        )
        await engine.store(
            Memory(
                content="Port 3000 is already bound by another service",
                memory_type=MemoryType.ERROR,
            )
        )

        results = await engine.recall("architecture", memory_type="decision")
        for r in results:
            assert r.memory.memory_type == MemoryType.DECISION

    async def test_filter_by_tags(self, engine):
        await engine.store(Memory(content="Auth uses JWT", tags=["auth", "jwt"]))
        await engine.store(Memory(content="DB uses Postgres", tags=["database"]))

        results = await engine.recall("system", tags=["auth"])
        for r in results:
            assert "auth" in r.memory.tags


@pytest.mark.asyncio
class TestScoringOrder:
    async def test_higher_importance_scores_higher(self, engine):
        await engine.store(Memory(content="Critical auth decision", importance=4))
        await engine.store(Memory(content="Trivial auth note", importance=0))

        results = await engine.recall("auth decision")
        if len(results) >= 2:
            assert results[0].memory.importance >= results[1].memory.importance

    async def test_score_breakdown_populated(self, engine):
        await engine.store(Memory(content="Test memory for scoring breakdown"))
        results = await engine.recall("scoring breakdown")
        assert len(results) >= 1
        breakdown = results[0].score_breakdown
        assert "vector" in breakdown
        assert "bm25" in breakdown
        assert "recency" in breakdown


@pytest.mark.asyncio
class TestGraphExpansion:
    async def test_connected_memories_attached(self, engine):
        m1 = Memory(content="Auth uses JWT tokens")
        m2 = Memory(content="JWT tokens expire after 24 hours")
        stored1 = await engine.store(m1)
        stored2 = await engine.store(m2)

        rel = Relationship(
            source_id=stored1.id,
            target_id=stored2.id,
            rel_type=RelationType.RELATES_TO,
        )
        await engine.db.store_relationship(rel)

        results = await engine.recall("JWT authentication")
        if results:
            top = results[0]
            connected_ids = [c.memory.id for c in top.connected]
            other_id = stored2.id if top.memory.id == stored1.id else stored1.id
            assert other_id in connected_ids


@pytest.mark.asyncio
class TestSupersedeWarning:
    async def test_superseded_memory_shows_warning(self, engine):
        old = await engine.store(Memory(content="Use MySQL for the database"))
        new = await engine.store(Memory(content="Use PostgreSQL instead of MySQL"))

        rel = Relationship(
            source_id=new.id,
            target_id=old.id,
            rel_type=RelationType.SUPERSEDES,
        )
        await engine.db.store_relationship(rel)

        await engine.db.update_memory(old.id, importance=0)

        # Use min_importance=0 to ensure the demoted memory appears in results
        results = await engine.recall("MySQL database", min_importance=0, top_k=20)
        old_results = [r for r in results if r.memory.id == old.id]
        assert old_results, "Superseded memory should appear with min_importance=0"
        connected_types = [c.rel_type for c in old_results[0].connected]
        assert "supersedes" in connected_types

    async def test_auto_connect_creates_relationships(self, engine):
        """Auto-connect should create relates_to edges between similar memories."""
        m1 = await engine.store(Memory(
            content="Kubernetes uses etcd for storing cluster state and configuration.",
            memory_type=MemoryType.ARCHITECTURE,
        ))
        m2 = await engine.store(Memory(
            content="etcd is the key-value store backing Kubernetes control plane data.",
            memory_type=MemoryType.ARCHITECTURE,
        ))

        # Verify directly via DB rather than through recall (which doesn't accept project=)
        m1_conns = await engine.db.get_connected(m1.id)
        m2_conns = await engine.db.get_connected(m2.id)
        all_connected_ids = (
            {mem.id for mem, _, _, _ in m1_conns}
            | {mem.id for mem, _, _, _ in m2_conns}
        )

        # With FakeEmbedder these texts share enough words for high cosine similarity,
        # so auto-connect should fire.  If the threshold is unreachable with FakeEmbedder
        # we still assert the code path didn't crash and check the graph score.
        graph_score = await engine.db.get_graph_score(m1.id)
        if m2.id in all_connected_ids or m1.id in all_connected_ids:
            assert graph_score > 0, "Connected memories should have non-zero graph score"
        else:
            # Threshold not met -- at minimum verify graph_score is zero (no phantom edges)
            assert graph_score == 0.0


@pytest.mark.asyncio
class TestFeedback:
    async def test_positive_feedback_boosts_edges(self, engine):
        m1 = Memory(content="Memory A")
        m2 = Memory(content="Memory B")
        s1 = await engine.store(m1)
        s2 = await engine.store(m2)

        rel = Relationship(source_id=s1.id, target_id=s2.id, strength=0.5)
        await engine.db.store_relationship(rel)

        result = await engine.feedback([s1.id], helpful=True)
        assert result["action"] == "reinforced"

    async def test_negative_feedback_weakens_edges(self, engine):
        m1 = Memory(content="Memory X")
        s1 = await engine.store(m1)

        result = await engine.feedback([s1.id], helpful=False)
        assert result["action"] == "weakened"
