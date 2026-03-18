from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from .db import MemoryDB
from .embeddings import EmbeddingClient, cosine_similarity, from_blob, to_blob
from .chunker import chunk_hash, chunk_text, is_duplicate
from .types import (
    Chunk,
    ConnectedMemory,
    Memory,
    SearchResult,
)

WEIGHT_VECTOR = 0.45
WEIGHT_BM25 = 0.25
WEIGHT_RECENCY = 0.15
WEIGHT_GRAPH = 0.15

DECAY_RATE = 0.01  # per hour


class SearchEngine:
    def __init__(self, db: MemoryDB, embedder: EmbeddingClient):
        self.db = db
        self.embedder = embedder

    def store(self, memory: Memory) -> Memory:
        """Store a memory: chunk it, embed it, index it."""
        memory = self.db.store_memory(memory)

        chunks = chunk_text(memory.content)
        existing_texts = [
            c.chunk_text
            for c in self.db.get_all_chunks_with_embeddings(limit=5000)
        ]

        texts_to_embed: list[str] = []
        chunk_objects: list[Chunk] = []

        for i, text in enumerate(chunks):
            h = chunk_hash(text)
            if self.db.chunk_hash_exists(h):
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

        if texts_to_embed:
            embeddings = self.embedder.embed_batch(texts_to_embed)
            for chunk_obj, emb in zip(chunk_objects, embeddings):
                chunk_obj.embedding = to_blob(emb)

        if chunk_objects:
            self.db.store_chunks(chunk_objects)

        return memory

    def recall(
        self,
        query: str,
        top_k: int = 10,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        min_importance: int | None = None,
        graph_hops: int = 1,
    ) -> list[SearchResult]:
        """Three-layer recall: BM25 + vector + recency, then graph expansion."""

        candidates: dict[str, _Candidate] = {}

        # Layer 1: FTS5 / BM25
        fts_results = self.db.fts_search(query, limit=top_k * 2)
        if fts_results:
            max_bm25 = max(score for _, score in fts_results) or 1.0
            for mem, score in fts_results:
                norm_score = score / max_bm25
                cand = candidates.setdefault(mem.id, _Candidate(memory=mem))
                cand.bm25_score = norm_score
                cand.matched_chunk = mem.content[:200]

        # Layer 2: Vector / Semantic
        query_vec = self.embedder.embed(query)
        all_chunks = self.db.get_all_chunks_with_embeddings()

        chunk_scores: list[tuple[Chunk, float]] = []
        for chunk in all_chunks:
            if chunk.embedding is None:
                continue
            chunk_vec = from_blob(chunk.embedding)
            sim = cosine_similarity(query_vec, chunk_vec)
            chunk_scores.append((chunk, sim))

        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[: top_k * 2]

        if top_chunks:
            max_vec = top_chunks[0][1] if top_chunks[0][1] > 0 else 1.0
            for chunk, sim in top_chunks:
                norm_score = sim / max_vec if max_vec > 0 else 0
                mem = self.db.get_memory(chunk.memory_id)
                if not mem:
                    continue
                cand = candidates.setdefault(mem.id, _Candidate(memory=mem))
                if norm_score > cand.vector_score:
                    cand.vector_score = norm_score
                    cand.matched_chunk = chunk.chunk_text[:200]

        # Score each candidate
        now = datetime.now(timezone.utc)
        scored: list[SearchResult] = []

        for cand in candidates.values():
            mem = cand.memory

            # Apply filters
            if memory_type and mem.memory_type.value != memory_type:
                continue
            if min_importance is not None and mem.importance > min_importance:
                continue
            if tags and not (set(tags) & set(mem.tags)):
                continue

            # Layer 3: Recency decay
            hours = max((now - mem.last_accessed).total_seconds() / 3600, 0.01)
            recency_score = math.exp(-DECAY_RATE * hours)

            # Layer 4: Graph connectivity boost (Cognee-inspired)
            conn_count = self.db.get_connection_count(mem.id)
            graph_score = min(1.0, conn_count / 5.0)

            composite = (
                WEIGHT_VECTOR * cand.vector_score
                + WEIGHT_BM25 * cand.bm25_score
                + WEIGHT_RECENCY * recency_score
                + WEIGHT_GRAPH * graph_score
            )

            # Importance multiplier: importance 0 => 2x, importance 4 => 0.6x
            importance_mult = 2.0 - (mem.importance * 0.35)
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

        # Graph expansion: attach connected memories
        for result in top_results:
            self.db.touch_memory(result.memory.id)
            connected_raw = self.db.get_connected(result.memory.id, max_hops=graph_hops)
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

    def feedback(self, memory_ids: list[str], helpful: bool) -> dict:
        """Reinforce or weaken graph edges connected to recalled memories.

        Inspired by Cognee's feedback loop: when results are helpful,
        the graph paths that produced them get stronger. When unhelpful,
        they get weaker. Over time, the graph self-optimizes.
        """
        total_affected = 0
        boost = 0.05 if helpful else -0.05

        for mid in memory_ids:
            mem = self.db.get_memory(mid)
            if not mem:
                continue
            if helpful:
                self.db.boost_edges_for_memory(mid, abs(boost))
                self.db.touch_memory(mid)
            else:
                self.db.decay_edges_for_memory(mid, abs(boost))
            total_affected += 1

        return {
            "action": "reinforced" if helpful else "weakened",
            "memories_affected": total_affected,
        }

    def memify(self) -> dict:
        """Memory enhancement pass -- Cognee's memify concept.

        Three stages:
        1. Deduplicate chunks (by hash)
        2. Decay all edge strengths and prune weak edges
        3. Prune stale, never-accessed, low-importance memories
        """
        # Stage 1: Dedup chunks
        deduped = self._dedup_chunks()

        # Stage 2: Decay and prune edges
        decayed, pruned_edges = self.db.decay_all_edges(decay_factor=0.02, min_strength=0.1)

        # Stage 3: Prune stale memories (30 days, low importance, never accessed)
        pruned_memories = self.db.prune_stale_memories(max_age_hours=720, max_importance=3)

        return {
            "chunks_deduped": deduped,
            "edges_decayed": decayed,
            "edges_pruned": pruned_edges,
            "stale_memories_pruned": pruned_memories,
        }

    def _dedup_chunks(self) -> int:
        all_chunks = self.db.get_all_chunks_with_embeddings()
        merged = 0
        seen_hashes: set[str] = set()
        for chunk in all_chunks:
            h = chunk.chunk_hash
            if h in seen_hashes:
                self.db._get_conn().execute("DELETE FROM chunks WHERE id = ?", (chunk.id,))
                merged += 1
            else:
                seen_hashes.add(h)
        if merged:
            self.db._get_conn().commit()
        return merged


class _Candidate:
    __slots__ = ("memory", "bm25_score", "vector_score", "matched_chunk")

    def __init__(self, memory: Memory):
        self.memory = memory
        self.bm25_score: float = 0.0
        self.vector_score: float = 0.0
        self.matched_chunk: str = ""
