"""Smoke tests for MCP tool functions (async Postgres)."""

from __future__ import annotations

import pytest
import pytest_asyncio

from tests.conftest import FakeEmbedder


@pytest.fixture(autouse=True)
def _clear_engines():
    import engram.server as srv

    srv._engines.clear()
    yield
    srv._engines.clear()


@pytest_asyncio.fixture(autouse=True)
async def _server_base(pg_pool, monkeypatch):
    import engram.server as srv

    async def _return_pool():
        return pg_pool

    monkeypatch.setattr(srv, "get_pool", _return_pool)
    monkeypatch.setenv("ENGRAM_EMBEDDER", "none")
    monkeypatch.delenv("ENGRAM_ALLOWED_PROJECTS", raising=False)

    async with pg_pool.connection() as conn:
        await conn.execute("TRUNCATE memories, project_meta CASCADE")
    yield


@pytest_asyncio.fixture
async def _patch_embedder(monkeypatch, pg_pool):
    import engram.server as srv
    from engram.db import MemoryDB
    from engram.search import SearchEngine
    from engram.util import normalize_project

    async def patched_get_engine(project=None):
        proj = normalize_project(project)
        if proj not in srv._engines:
            db = MemoryDB(project=proj, pool=pg_pool)
            embedder = FakeEmbedder()
            srv._engines[proj] = SearchEngine(db=db, embedder=embedder)
            srv._engines.move_to_end(proj)
        return srv._engines[proj]

    monkeypatch.setattr(srv, "_get_engine", patched_get_engine)


@pytest.mark.asyncio
class TestMemoryStoreRecall:
    async def test_store_and_recall(self, _patch_embedder):
        from engram.server import memory_recall, memory_store

        result = await memory_store(
            content="We use PostgreSQL for the main database",
            memory_type="decision",
            tags="database,postgres",
            importance=3,
            project="test-project",
        )
        assert result["status"] == "stored"
        assert result["memory_type"] == "decision"

        recall = await memory_recall(
            query="database choice",
            project="test-project",
        )
        assert recall["count"] >= 1
        assert "PostgreSQL" in recall["results"][0]["content"]

    async def test_project_isolation(self, _patch_embedder):
        from engram.server import memory_recall, memory_store

        await memory_store(
            content="Alpha project secret",
            project="alpha",
        )
        recall = await memory_recall(query="secret", project="beta")
        assert recall["count"] == 0


@pytest.mark.asyncio
class TestMemoryCorrect:
    async def test_correct_supersedes_old(self, _patch_embedder):
        from engram.server import memory_correct, memory_store

        store_result = await memory_store(
            content="Use MySQL for the database",
            memory_type="decision",
            tags="database",
            project="test-project",
        )
        old_id = store_result["id"]

        correct_result = await memory_correct(
            old_memory_id=old_id,
            new_content="Use PostgreSQL instead of MySQL for JSONB support",
            project="test-project",
        )
        assert correct_result["status"] == "corrected"
        assert "trivial" in correct_result["old_demoted_to"]

    async def test_correct_shows_warning_on_recall(self, _patch_embedder):
        """Superseded memories must include WARNING + superseded_by in recall output."""
        from engram.server import memory_correct, memory_recall, memory_store

        store_result = await memory_store(
            content="Use MySQL for the database",
            memory_type="decision",
            tags="database",
            project="test-project",
        )
        old_id = store_result["id"]

        correct_result = await memory_correct(
            old_memory_id=old_id,
            new_content="Use PostgreSQL instead of MySQL for JSONB support",
            project="test-project",
        )
        new_id = correct_result["new_id"]

        # Use min_importance=0 so the demoted (importance=0) memory appears in results
        recall = await memory_recall(
            query="MySQL database",
            min_importance=0,
            top_k=20,
            project="test-project",
        )

        old_entries = [r for r in recall["results"] if r["id"] == old_id]
        assert old_entries, (
            f"Superseded memory {old_id} should appear in recall with min_importance=0"
        )
        entry = old_entries[0]
        assert "WARNING" in entry, "Superseded memory should have WARNING field"
        assert "superseded_by" in entry, "Should include superseded_by reference"
        assert "THIS MEMORY HAS BEEN SUPERSEDED" in entry["WARNING"]
        assert entry["superseded_by"]["id"] == new_id

    async def test_correct_nonexistent_raises(self, _patch_embedder):
        from engram.server import memory_correct

        with pytest.raises(ValueError, match="not found"):
            await memory_correct(
                old_memory_id="nonexistent",
                new_content="Doesn't matter",
                project="test-project",
            )


@pytest.mark.asyncio
class TestMemoryForget:
    async def test_forget_removes_memory(self, _patch_embedder):
        from engram.server import memory_forget, memory_store

        store_result = await memory_store(content="Delete me", project="test-project")
        mid = store_result["id"]

        forget_result = await memory_forget(memory_id=mid, project="test-project")
        assert forget_result["status"] == "forgotten"

    async def test_forget_nonexistent_raises(self, _patch_embedder):
        from engram.server import memory_forget

        with pytest.raises(ValueError, match="not found"):
            await memory_forget(memory_id="nope", project="test-project")


@pytest.mark.asyncio
class TestMemoryList:
    async def test_list_returns_stored_memories(self, _patch_embedder):
        from engram.server import memory_list, memory_store

        await memory_store(content="First memory", project="test-project")
        await memory_store(content="Second memory", project="test-project")

        result = await memory_list(project="test-project")
        assert result["count"] == 2


@pytest.mark.asyncio
class TestMemoryStatus:
    async def test_status_returns_stats(self, _patch_embedder):
        from engram.server import memory_status, memory_store

        await memory_store(content="A memory", project="test-project")
        stats = await memory_status(project="test-project")
        assert stats["total_memories"] == 1
