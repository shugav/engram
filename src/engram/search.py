"""Three-layer memory search (BM25 + pgvector + graph) for Engram v3."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from .chunker import chunk_hash, chunk_text, is_duplicate
from .db import MemoryDB
from .embeddings import (
    EMBEDDING_DIM,
    EmbeddingProvider,
    NullEmbedder,
    format_vector_literal,
    to_blob,
)
from .errors import EmbeddingConfigMismatchError
from .types import (
    Chunk,
    ConnectedMemory,
    Memory,
    Relationship,
    RelationType,
    SearchResult,
)

logger = logging.getLogger(__name__)

WEIGHT_VECTOR = 0.45
WEIGHT_BM25 = 0.25
WEIGHT_RECENCY = 0.15
WEIGHT_GRAPH = 0.15

# When vectors are disabled, redistribute former vector mass to BM25 + recency (+ graph).
WEIGHT_BM25_NULL = 0.50
WEIGHT_RECENCY_NULL = 0.30
WEIGHT_GRAPH_NULL = 0.20

DECAY_RATE = 0.01  # per hour


class SearchEngine:
    def __init__(self, db: MemoryDB, embedder: EmbeddingProvider):
        self.db = db
        self.embedder = embedder
        self._is_null = isinstance(embedder, NullEmbedder)

    @property
    def has_vectors(self) -> bool:
        return not self._is_null

    def _weights(self) -> tuple[float, float, float, float]:
        if self._is_null:
            return 0.0, WEIGHT_BM25_NULL, WEIGHT_RECENCY_NULL, WEIGHT_GRAPH_NULL
        return WEIGHT_VECTOR, WEIGHT_BM25, WEIGHT_RECENCY, WEIGHT_GRAPH

    async def _check_embedder_metadata(self) -> None:
        if self._is_null:
            return

        stored_name = await self.db.get_meta("embedder_name")
        stored_dims = await self.db.get_meta("embedder_dimensions")

        if stored_name is None:
            await self.db.set_meta("embedder_name", self.embedder.name)
            await self.db.set_meta("embedder_dimensions", str(EMBEDDING_DIM))
            await self.db.set_meta("embedder_version", getattr(self.embedder, "version", "unknown"))
            return

        if stored_name != self.embedder.name or int(stored_dims or 0) != EMBEDDING_DIM:
            raise EmbeddingConfigMismatchError(
                stored_name=stored_name,
                stored_dims=int(stored_dims or 0),
                current_name=self.embedder.name,
                current_dims=EMBEDDING_DIM,
            )

    async def _prepare_new_chunks(self, memory: Memory) -> tuple[list[Chunk], list[str]]:
        """Build deduped chunk rows and parallel text list for embedding."""
        chunks = chunk_text(memory.content)
        existing_texts = await self.db.get_all_chunk_texts(limit=5000)
        texts_to_embed: list[str] = []
        chunk_objects: list[Chunk] = []
        for i, text in enumerate(chunks):
            h = chunk_hash(text)
            if await self.db.chunk_hash_exists(h):
                continue
            if is_duplicate(text, existing_texts):
                continue
            chunk_objects.append(
                Chunk(
                    memory_id=memory.id,
                    chunk_text=text,
                    chunk_index=i,
                    chunk_hash=h,
                )
            )
            texts_to_embed.append(text)
            existing_texts.append(text)
        return chunk_objects, texts_to_embed

    async def store(self, memory: Memory) -> Memory:
        await self._check_embedder_metadata()
        chunk_objects, texts_to_embed = await self._prepare_new_chunks(memory)

        async with self.db.pool.connection() as conn:
            async with conn.transaction():
                memory = await self.db.store_memory(memory, conn=conn)

                if texts_to_embed and self.has_vectors:
                    try:
                        embeddings = await self.embedder.embed_batch(texts_to_embed)
                    except Exception:
                        logger.exception("Embedding failed; rolling back memory store")
                        raise
                    for chunk_obj, emb in zip(chunk_objects, embeddings):
                        chunk_obj.embedding = to_blob(emb)

                if chunk_objects:
                    await self.db.store_chunks(chunk_objects, conn=conn)

        return memory

    async def correct_memory(
        self,
        old_memory_id: str,
        new_memory: Memory,
    ) -> tuple[Memory, Memory]:
        """Store correction, supersede link, and demote old memory (single transaction)."""
        await self._check_embedder_metadata()
        old = await self.db.get_memory(old_memory_id)
        if not old:
            raise ValueError(f"Memory '{old_memory_id}' not found.")

        chunk_objects, texts_to_embed = await self._prepare_new_chunks(new_memory)

        async with self.db.pool.connection() as conn:
            async with conn.transaction():
                stored_new = await self.db.store_memory(new_memory, conn=conn)

                if texts_to_embed and self.has_vectors:
                    try:
                        embeddings = await self.embedder.embed_batch(texts_to_embed)
                    except Exception:
                        logger.exception("Embedding failed; rolling back correction")
                        raise
                    for chunk_obj, emb in zip(chunk_objects, embeddings):
                        chunk_obj.memory_id = stored_new.id
                        chunk_obj.embedding = to_blob(emb)
                else:
                    for chunk_obj in chunk_objects:
                        chunk_obj.memory_id = stored_new.id

                if chunk_objects:
                    await self.db.store_chunks(chunk_objects, conn=conn)

                rel = Relationship(
                    source_id=stored_new.id,
                    target_id=old_memory_id,
                    rel_type=RelationType.SUPERSEDES,
                    strength=1.0,
                )
                await self.db.store_relationship(rel, conn=conn)
                await self.db.update_memory(old_memory_id, importance=0, conn=conn)

        return old, stored_new

    async def recall(
        self,
        query: str,
        top_k: int = 10,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        min_importance: int | None = None,
        graph_hops: int = 1,
    ) -> list[SearchResult]:
        w_vec, w_bm25, w_rec, w_graph = self._weights()
        candidates: dict[str, _Candidate] = {}

        fts_results = await self.db.fts_search(query, limit=top_k * 2)
        if fts_results:
            max_bm25 = max(score for _, score in fts_results) or 1.0
            for mem, score in fts_results:
                norm_score = score / max_bm25
                cand = candidates.setdefault(mem.id, _Candidate(memory=mem))
                cand.bm25_score = norm_score
                cand.matched_chunk = mem.content[:200]

        if self.has_vectors:
            query_vec = await self.embedder.embed(query)
            qv = format_vector_literal(query_vec)
            nearest = await self.db.nearest_chunks_by_embedding(qv, limit=top_k * 2)
            if nearest:
                max_vec = max(sim for _, sim in nearest) or 1.0
                for chunk, sim in nearest:
                    norm_score = sim / max_vec if max_vec > 0 else 0.0
                    mem = await self.db.get_memory(chunk.memory_id)
                    if not mem:
                        continue
                    cand = candidates.setdefault(mem.id, _Candidate(memory=mem))
                    if norm_score > cand.vector_score:
                        cand.vector_score = norm_score
                        cand.matched_chunk = chunk.chunk_text[:200]

        now = datetime.now(timezone.utc)
        scored: list[SearchResult] = []

        for cand in candidates.values():
            mem = cand.memory

            if memory_type and mem.memory_type.value != memory_type:
                continue
            if min_importance is not None and mem.importance < min_importance:
                continue
            if tags and not (set(tags) & set(mem.tags)):
                continue

            hours = max((now - mem.last_accessed).total_seconds() / 3600, 0.01)
            recency_score = math.exp(-DECAY_RATE * hours)

            conn_count = await self.db.get_connection_count(mem.id)
            graph_score = min(1.0, conn_count / 5.0)

            composite = (
                w_vec * cand.vector_score
                + w_bm25 * cand.bm25_score
                + w_rec * recency_score
                + w_graph * graph_score
            )

            importance_mult = 1.0 + (mem.importance * 0.125)
            final_score = composite * importance_mult

            scored.append(
                SearchResult(
                    memory=mem,
                    score=round(final_score, 4),
                    score_breakdown={
                        "vector": round(cand.vector_score, 4),
                        "bm25": round(cand.bm25_score, 4),
                        "recency": round(recency_score, 4),
                        "graph": round(graph_score, 4),
                        "importance_mult": round(importance_mult, 2),
                    },
                    matched_chunk=cand.matched_chunk,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        top_results = scored[:top_k]

        for result in top_results:
            await self.db.touch_memory(result.memory.id)
            connected_raw = await self.db.get_connected(result.memory.id, max_hops=graph_hops)
            result.connected = [
                ConnectedMemory(
                    memory=mem,
                    rel_type=rel_type,
                    direction=direction,
                    strength=strength,
                )
                for mem, rel_type, direction, strength in connected_raw
            ]

        return top_results

    async def feedback(self, memory_ids: list[str], helpful: bool) -> dict:
        total_affected = 0
        boost = 0.05 if helpful else -0.05

        for mid in memory_ids:
            mem = await self.db.get_memory(mid)
            if not mem:
                continue
            if helpful:
                await self.db.boost_edges_for_memory(mid, abs(boost))
                await self.db.touch_memory(mid)
            else:
                await self.db.decay_edges_for_memory(mid, abs(boost))
            total_affected += 1

        return {
            "action": "reinforced" if helpful else "weakened",
            "memories_affected": total_affected,
        }

    async def memify(self) -> dict:
        deduped = await self._dedup_chunks()
        decayed, pruned_edges = await self.db.decay_all_edges(decay_factor=0.02, min_strength=0.1)
        pruned_memories = await self.db.prune_stale_memories(max_age_hours=720, max_importance=1)

        return {
            "chunks_deduped": deduped,
            "edges_decayed": decayed,
            "edges_pruned": pruned_edges,
            "stale_memories_pruned": pruned_memories,
        }

    async def _dedup_chunks(self) -> int:
        all_chunks = await self.db.get_all_chunks_with_embeddings()
        merged = 0
        seen_hashes: set[str] = set()
        dup_ids: list[str] = []
        for chunk in all_chunks:
            h = chunk.chunk_hash
            if h in seen_hashes:
                dup_ids.append(chunk.id)
                merged += 1
            else:
                seen_hashes.add(h)
        if dup_ids:
            await self.db.delete_chunk_ids(dup_ids)
        return merged


class _Candidate:
    __slots__ = ("memory", "bm25_score", "vector_score", "matched_chunk")

    def __init__(self, memory: Memory):
        self.memory = memory
        self.bm25_score = 0.0
        self.vector_score = 0.0
        self.matched_chunk = ""
