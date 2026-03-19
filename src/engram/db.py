"""Postgres-backed memory database (Engram v3).

Purpose: CRUD for memories, chunks, relationships, FTS, and stats per logical project.
Role: All data lives in one database; ``project`` column isolates tenants.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Sequence

from psycopg.rows import dict_row
from psycopg.types.json import Json

from .types import (
    Chunk,
    Memory,
    MemoryStats,
    MemoryType,
    Relationship,
)
from .util import normalize_project

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(v: Any) -> str:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


class MemoryDB:
    """Async Postgres access for one logical ``project`` (shared pool)."""

    def __init__(self, project: str, pool: AsyncConnectionPool) -> None:
        self.project = normalize_project(project)
        self._pool = pool

    async def close(self) -> None:
        """No-op per instance (pool is shared)."""

    @property
    def pool(self) -> AsyncConnectionPool:
        """Expose pool for tests and transactional orchestration."""
        return self._pool

    # ── connection helper ────────────────────────────────────────

    @asynccontextmanager
    async def _acquire(self, outer: AsyncConnection | None):
        if outer is not None:
            yield outer
            return
        async with self._pool.connection() as conn:
            yield conn

    # ── Project Metadata ─────────────────────────────────────────

    async def get_meta(self, key: str, conn: AsyncConnection | None = None) -> str | None:
        async with self._acquire(conn) as c:
            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT value FROM project_meta WHERE project = %s AND key = %s",
                    (self.project, key),
                )
                row = await cur.fetchone()
                return row["value"] if row else None

    async def set_meta(self, key: str, value: str, conn: AsyncConnection | None = None) -> None:
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """INSERT INTO project_meta (project, key, value)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (project, key) DO UPDATE SET value = EXCLUDED.value""",
                    (self.project, key, value),
                )

    # ── Memory CRUD ──────────────────────────────────────────────

    async def store_memory(self, memory: Memory, conn: AsyncConnection | None = None) -> Memory:
        now = _now_utc()
        memory.created_at = now
        memory.updated_at = now
        memory.last_accessed = now
        memory.project = self.project

        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """INSERT INTO memories (id, content, memory_type, project, tags,
                       importance, access_count, last_accessed, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        memory.id,
                        memory.content,
                        memory.memory_type.value,
                        memory.project,
                        Json(memory.tags),
                        memory.importance,
                        memory.access_count,
                        memory.last_accessed,
                        memory.created_at,
                        memory.updated_at,
                    ),
                )
        return memory

    async def get_memory(
        self, memory_id: str, conn: AsyncConnection | None = None
    ) -> Memory | None:
        async with self._acquire(conn) as c:
            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM memories WHERE id = %s AND project = %s",
                    (memory_id, self.project),
                )
                row = await cur.fetchone()
                if not row:
                    return None
                return self._row_to_memory(row)

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
        importance: int | None = None,
        conn: AsyncConnection | None = None,
    ) -> Memory | None:
        mem = await self.get_memory(memory_id, conn=conn)
        if not mem:
            return None
        if content is not None:
            mem.content = content
        if tags is not None:
            mem.tags = tags
        if importance is not None:
            mem.importance = importance
        now = _now_utc()
        mem.updated_at = now
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """UPDATE memories SET content=%s, tags=%s, importance=%s, updated_at=%s
                       WHERE id=%s AND project=%s""",
                    (mem.content, Json(mem.tags), mem.importance, now, memory_id, self.project),
                )
        return mem

    async def delete_memory(self, memory_id: str, conn: AsyncConnection | None = None) -> bool:
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    "DELETE FROM memories WHERE id = %s AND project = %s",
                    (memory_id, self.project),
                )
                return cur.rowcount > 0

    async def forget_memory(self, memory_id: str) -> bool:
        """Delete a memory and cascaded chunks/relationships (single statement)."""
        return await self.delete_memory(memory_id)

    async def list_memories(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        min_importance: int | None = None,
        limit: int = 20,
        offset: int = 0,
        conn: AsyncConnection | None = None,
    ) -> list[Memory]:
        """List memories; ``min_importance`` is a floor (importance >= value) when set."""
        q = "SELECT * FROM memories WHERE project = %s"
        params: list[Any] = [self.project]
        if memory_type:
            q += " AND memory_type = %s"
            params.append(memory_type.value)
        if min_importance is not None:
            q += " AND importance >= %s"
            params.append(min_importance)
        if tags:
            q += " AND tags ?| %s::text[]"
            params.append(tags)
        q += " ORDER BY updated_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        async with self._acquire(conn) as c:
            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(q, params)
                rows = await cur.fetchall()
                return [self._row_to_memory(r) for r in rows]

    async def touch_memory(self, memory_id: str, conn: AsyncConnection | None = None) -> None:
        now = _now_utc()
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """UPDATE memories SET access_count = access_count + 1, last_accessed = %s
                       WHERE id = %s AND project = %s""",
                    (now, memory_id, self.project),
                )

    # ── Chunk CRUD ───────────────────────────────────────────────

    async def store_chunks(
        self,
        chunks: list[Chunk],
        conn: AsyncConnection | None = None,
    ) -> None:
        from .embeddings import format_vector_literal

        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                for ch in chunks:
                    if ch.embedding is not None and len(ch.embedding) > 0:
                        lit = format_vector_literal(ch.embedding)
                        await cur.execute(
                            """INSERT INTO chunks (id, memory_id, chunk_text, chunk_index,
                               chunk_hash, embedding)
                               VALUES (%s, %s, %s, %s, %s, %s::vector)
                               ON CONFLICT (id) DO NOTHING""",
                            (
                                ch.id,
                                ch.memory_id,
                                ch.chunk_text,
                                ch.chunk_index,
                                ch.chunk_hash,
                                lit,
                            ),
                        )
                    else:
                        await cur.execute(
                            """INSERT INTO chunks (id, memory_id, chunk_text, chunk_index,
                               chunk_hash, embedding)
                               VALUES (%s, %s, %s, %s, %s, NULL)
                               ON CONFLICT (id) DO NOTHING""",
                            (
                                ch.id,
                                ch.memory_id,
                                ch.chunk_text,
                                ch.chunk_index,
                                ch.chunk_hash,
                            ),
                        )

    async def get_chunks_for_memory(
        self,
        memory_id: str,
        conn: AsyncConnection | None = None,
    ) -> list[Chunk]:
        async with self._acquire(conn) as c:
            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM chunks WHERE memory_id = %s ORDER BY chunk_index",
                    (memory_id,),
                )
                rows = await cur.fetchall()
                return [self._row_to_chunk(r) for r in rows]

    async def nearest_chunks_by_embedding(
        self,
        query_embedding: Any,
        limit: int,
        conn: AsyncConnection | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return (chunk, cosine_similarity) using pgvector ANN ordering."""
        async with self._acquire(conn) as c:
            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """SELECT c.*, (1 - (c.embedding <=> %s::vector)) AS sim
                       FROM chunks c
                       JOIN memories m ON m.id = c.memory_id
                       WHERE m.project = %s AND c.embedding IS NOT NULL
                       ORDER BY c.embedding <=> %s::vector
                       LIMIT %s""",
                    (query_embedding, self.project, query_embedding, limit),
                )
                rows = await cur.fetchall()
                out: list[tuple[Chunk, float]] = []
                for r in rows:
                    rowd = dict(r)
                    sim = float(rowd.pop("sim", 0.0))
                    out.append((self._row_to_chunk(rowd), sim))
                return out

    async def get_all_chunks_with_embeddings(
        self,
        limit: int = 10_000,
        conn: AsyncConnection | None = None,
    ) -> list[Chunk]:
        async with self._acquire(conn) as c:
            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """SELECT c.* FROM chunks c
                       JOIN memories m ON m.id = c.memory_id
                       WHERE c.embedding IS NOT NULL AND m.project = %s
                       ORDER BY m.last_accessed DESC
                       LIMIT %s""",
                    (self.project, limit),
                )
                rows = await cur.fetchall()
                return [self._row_to_chunk(r) for r in rows]

    async def get_all_chunk_texts(
        self,
        limit: int = 5000,
        conn: AsyncConnection | None = None,
    ) -> list[str]:
        async with self._acquire(conn) as c:
            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """SELECT c.chunk_text FROM chunks c
                       JOIN memories m ON m.id = c.memory_id
                       WHERE m.project = %s
                       LIMIT %s""",
                    (self.project, limit),
                )
                rows = await cur.fetchall()
                return [r["chunk_text"] for r in rows]

    async def chunk_hash_exists(
        self,
        chunk_hash: str,
        conn: AsyncConnection | None = None,
    ) -> bool:
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """SELECT 1 FROM chunks c
                       JOIN memories m ON m.id = c.memory_id
                       WHERE c.chunk_hash = %s AND m.project = %s LIMIT 1""",
                    (chunk_hash, self.project),
                )
                return await cur.fetchone() is not None

    async def delete_chunks_for_memory(
        self,
        memory_id: str,
        conn: AsyncConnection | None = None,
    ) -> None:
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute("DELETE FROM chunks WHERE memory_id = %s", (memory_id,))

    async def delete_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        conn: AsyncConnection | None = None,
    ) -> int:
        if not chunk_ids:
            return 0
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """DELETE FROM chunks WHERE id = ANY(%s::text[])
                       AND memory_id IN (SELECT id FROM memories WHERE project = %s)""",
                    (list(chunk_ids), self.project),
                )
                return cur.rowcount or 0

    # ── Relationship CRUD ───────────────────────────────────────

    async def store_relationship(
        self,
        rel: Relationship,
        conn: AsyncConnection | None = None,
    ) -> Relationship:
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """INSERT INTO relationships (id, source_id, target_id, rel_type,
                       strength, created_at) VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (source_id, target_id, rel_type)
                       DO UPDATE SET strength = EXCLUDED.strength""",
                    (
                        rel.id,
                        rel.source_id,
                        rel.target_id,
                        rel.rel_type.value,
                        rel.strength,
                        rel.created_at,
                    ),
                )
        return rel

    async def get_connected(
        self,
        memory_id: str,
        max_hops: int = 2,
        conn: AsyncConnection | None = None,
    ) -> list[tuple[Memory, str, str, float]]:
        async with self._acquire(conn) as c:
            return await self._get_connected_with_conn(c, memory_id, max_hops)

    async def _get_connected_with_conn(
        self,
        c: AsyncConnection,
        memory_id: str,
        max_hops: int,
    ) -> list[tuple[Memory, str, str, float]]:
        visited: set[str] = {memory_id}
        results: list[tuple[Memory, str, str, float]] = []
        frontier = [memory_id]

        for _ in range(max_hops):
            if not frontier:
                break
            next_frontier: list[str] = []

            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """SELECT r.target_id, r.rel_type, r.strength
                       FROM relationships r WHERE r.source_id = ANY(%s::text[])""",
                    (frontier,),
                )
                outgoing = await cur.fetchall()

            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """SELECT r.source_id, r.rel_type, r.strength
                       FROM relationships r WHERE r.target_id = ANY(%s::text[])""",
                    (frontier,),
                )
                incoming = await cur.fetchall()

            new_ids: list[str] = []
            pending: list[tuple[str, str, str, float]] = []

            for row in outgoing:
                nid = row["target_id"]
                if nid not in visited:
                    visited.add(nid)
                    pending.append((nid, row["rel_type"], "outgoing", float(row["strength"])))
                    new_ids.append(nid)
                    next_frontier.append(nid)

            for row in incoming:
                nid = row["source_id"]
                if nid not in visited:
                    visited.add(nid)
                    pending.append((nid, row["rel_type"], "incoming", float(row["strength"])))
                    new_ids.append(nid)
                    next_frontier.append(nid)

            if new_ids:
                mem_map = await self._fetch_memories_map(new_ids, c)
                for nid, rtype, direction, strength in pending:
                    mem = mem_map.get(nid)
                    if mem:
                        results.append((mem, rtype, direction, strength))

            frontier = next_frontier

        return results

    async def _fetch_memories_map(
        self,
        ids: list[str],
        c: AsyncConnection,
    ) -> dict[str, Memory]:
        if not ids:
            return {}
        async with c.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM memories WHERE id = ANY(%s::text[]) AND project = %s",
                (ids, self.project),
            )
            rows = await cur.fetchall()
            return {r["id"]: self._row_to_memory(r) for r in rows}

    async def boost_edges_for_memory(
        self,
        memory_id: str,
        factor: float = 0.05,
        conn: AsyncConnection | None = None,
    ) -> int:
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """UPDATE relationships r SET strength = LEAST(1.0, r.strength + %s)
                       WHERE (r.source_id = %s OR r.target_id = %s)
                       AND EXISTS (SELECT 1 FROM memories m WHERE m.id = r.source_id AND m.project = %s)
                       AND EXISTS (SELECT 1 FROM memories m2 WHERE m2.id = r.target_id AND m2.project = %s)""",
                    (factor, memory_id, memory_id, self.project, self.project),
                )
                return cur.rowcount or 0

    async def decay_edges_for_memory(
        self,
        memory_id: str,
        factor: float = 0.05,
        conn: AsyncConnection | None = None,
    ) -> int:
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """UPDATE relationships r SET strength = GREATEST(0.0, r.strength - %s)
                       WHERE (r.source_id = %s OR r.target_id = %s)
                       AND EXISTS (SELECT 1 FROM memories m WHERE m.id = r.source_id AND m.project = %s)
                       AND EXISTS (SELECT 1 FROM memories m2 WHERE m2.id = r.target_id AND m2.project = %s)""",
                    (factor, memory_id, memory_id, self.project, self.project),
                )
                return cur.rowcount or 0

    async def get_connection_count(
        self,
        memory_id: str,
        conn: AsyncConnection | None = None,
    ) -> int:
        async with self._acquire(conn) as c:
            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """SELECT COUNT(*) AS c FROM relationships r
                       WHERE (r.source_id = %s OR r.target_id = %s)
                       AND EXISTS (SELECT 1 FROM memories m WHERE m.id = r.source_id AND m.project = %s)
                       AND EXISTS (SELECT 1 FROM memories m2 WHERE m2.id = r.target_id AND m2.project = %s)""",
                    (memory_id, memory_id, self.project, self.project),
                )
                row = await cur.fetchone()
                return int(row["c"]) if row else 0

    async def decay_all_edges(
        self,
        decay_factor: float = 0.02,
        min_strength: float = 0.1,
        conn: AsyncConnection | None = None,
    ) -> tuple[int, int]:
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """UPDATE relationships r SET strength = GREATEST(0.0, r.strength - %s)
                       WHERE EXISTS (SELECT 1 FROM memories m WHERE m.id = r.source_id AND m.project = %s)
                       AND EXISTS (SELECT 1 FROM memories m2 WHERE m2.id = r.target_id AND m2.project = %s)""",
                    (decay_factor, self.project, self.project),
                )
                decayed = cur.rowcount or 0
            async with c.cursor() as cur:
                await cur.execute(
                    """DELETE FROM relationships r
                       WHERE r.strength < %s
                       AND EXISTS (SELECT 1 FROM memories m WHERE m.id = r.source_id AND m.project = %s)
                       AND EXISTS (SELECT 1 FROM memories m2 WHERE m2.id = r.target_id AND m2.project = %s)""",
                    (min_strength, self.project, self.project),
                )
                pruned = cur.rowcount or 0
        return decayed, pruned

    async def prune_stale_memories(
        self,
        max_age_hours: float = 720,
        max_importance: int = 1,
        conn: AsyncConnection | None = None,
    ) -> int:
        """Prune stale never-accessed memories with importance <= max_importance."""
        cutoff = _now_utc() - timedelta(hours=max_age_hours)
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    """DELETE FROM memories
                       WHERE project = %s AND importance <= %s
                       AND last_accessed < %s AND access_count = 0""",
                    (self.project, max_importance, cutoff),
                )
                return cur.rowcount or 0

    async def delete_relationships_for_memory(
        self,
        memory_id: str,
        conn: AsyncConnection | None = None,
    ) -> None:
        async with self._acquire(conn) as c:
            async with c.cursor() as cur:
                await cur.execute(
                    "DELETE FROM relationships WHERE source_id = %s OR target_id = %s",
                    (memory_id, memory_id),
                )

    async def fts_search(
        self,
        query: str,
        limit: int = 20,
        conn: AsyncConnection | None = None,
    ) -> list[tuple[Memory, float]]:
        q = (query or "").strip()
        if not q:
            return []
        async with self._acquire(conn) as c:
            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """SELECT m.*, ts_rank_cd(m.content_tsv, plainto_tsquery('simple', %s)) AS rank
                       FROM memories m
                       WHERE m.project = %s
                       AND m.content_tsv @@ plainto_tsquery('simple', %s)
                       ORDER BY rank DESC
                       LIMIT %s""",
                    (q, self.project, q, limit),
                )
                rows = await cur.fetchall()
                results: list[tuple[Memory, float]] = []
                for row in rows:
                    rd = dict(row)
                    rank = float(rd.pop("rank", 0.0))
                    results.append((self._row_to_memory(rd), rank))
                return results

    async def get_stats(self, conn: AsyncConnection | None = None) -> MemoryStats:
        async with self._acquire(conn) as c:
            async with c.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT COUNT(*) AS c FROM memories WHERE project = %s",
                    (self.project,),
                )
                total = (await cur.fetchone())["c"]

                await cur.execute(
                    """SELECT COUNT(*) AS c FROM chunks c
                       JOIN memories m ON m.id = c.memory_id WHERE m.project = %s""",
                    (self.project,),
                )
                total_chunks = (await cur.fetchone())["c"]

                await cur.execute(
                    """SELECT COUNT(*) AS c FROM relationships r
                       WHERE EXISTS (SELECT 1 FROM memories m WHERE m.id = r.source_id AND m.project = %s)
                       AND EXISTS (SELECT 1 FROM memories m2 WHERE m2.id = r.target_id AND m2.project = %s)""",
                    (self.project, self.project),
                )
                total_rels = (await cur.fetchone())["c"]

                await cur.execute(
                    """SELECT memory_type, COUNT(*) AS c FROM memories
                       WHERE project = %s GROUP BY memory_type""",
                    (self.project,),
                )
                type_rows = await cur.fetchall()
                by_type = {r["memory_type"]: r["c"] for r in type_rows}

                await cur.execute(
                    """SELECT importance, COUNT(*) AS c FROM memories
                       WHERE project = %s GROUP BY importance""",
                    (self.project,),
                )
                imp_rows = await cur.fetchall()
                by_importance = {str(r["importance"]): r["c"] for r in imp_rows}

                await cur.execute(
                    "SELECT MIN(created_at) AS v FROM memories WHERE project = %s",
                    (self.project,),
                )
                oldest_row = await cur.fetchone()
                await cur.execute(
                    "SELECT MAX(created_at) AS v FROM memories WHERE project = %s",
                    (self.project,),
                )
                newest_row = await cur.fetchone()

        approx_bytes = int(total) * 400 + int(total_chunks) * 800 + int(total_rels) * 120

        return MemoryStats(
            total_memories=total,
            total_chunks=total_chunks,
            total_relationships=total_rels,
            by_type=by_type,
            by_importance=by_importance,
            oldest=_iso(oldest_row["v"]) if oldest_row and oldest_row["v"] else None,
            newest=_iso(newest_row["v"]) if newest_row and newest_row["v"] else None,
            db_size_bytes=approx_bytes,
        )

    @staticmethod
    def _row_to_memory(row: dict[str, Any]) -> Memory:
        tags = row["tags"]
        if isinstance(tags, str):
            tags = json.loads(tags)
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            project=row["project"],
            tags=list(tags) if tags is not None else [],
            importance=int(row["importance"]),
            access_count=int(row["access_count"]),
            last_accessed=row["last_accessed"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_chunk(row: dict[str, Any]) -> Chunk:
        from .embeddings import vector_to_numpy_bytes

        emb = row.get("embedding")
        blob: bytes | None
        if emb is None:
            blob = None
        else:
            blob = vector_to_numpy_bytes(emb)
        return Chunk(
            id=row["id"],
            memory_id=row["memory_id"],
            chunk_text=row["chunk_text"],
            chunk_index=int(row["chunk_index"]),
            chunk_hash=row["chunk_hash"] or "",
            embedding=blob,
        )
