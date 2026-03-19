from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    DECISION = "decision"
    PATTERN = "pattern"
    ERROR = "error"
    CONTEXT = "context"
    ARCHITECTURE = "architecture"
    PREFERENCE = "preference"


class RelationType(str, Enum):
    CAUSED_BY = "caused_by"
    RELATES_TO = "relates_to"
    DEPENDS_ON = "depends_on"
    SUPERSEDES = "supersedes"
    USED_IN = "used_in"
    RESOLVED_BY = "resolved_by"


class Importance(int, Enum):
    """v3 scale: higher = more important (4 = critical, 0 = trivial)."""

    TRIVIAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


MAX_CONTENT_LENGTH = 50_000


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str = Field(..., max_length=MAX_CONTENT_LENGTH)
    memory_type: MemoryType = MemoryType.CONTEXT
    project: str = "default"
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=Importance.MEDIUM, ge=0, le=4)
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    memory_id: str
    chunk_text: str
    chunk_index: int
    chunk_hash: str = ""
    embedding: bytes | None = None


class Relationship(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source_id: str
    target_id: str
    rel_type: RelationType = RelationType.RELATES_TO
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SearchResult(BaseModel):
    memory: Memory
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    matched_chunk: str = ""
    connected: list[ConnectedMemory] = Field(default_factory=list)


class ConnectedMemory(BaseModel):
    memory: Memory
    rel_type: str
    direction: str  # "outgoing" or "incoming"
    strength: float = 1.0


class MemoryStats(BaseModel):
    total_memories: int = 0
    total_chunks: int = 0
    total_relationships: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    by_importance: dict[str, int] = Field(default_factory=dict)
    oldest: str | None = None
    newest: str | None = None
    db_size_bytes: int = 0


# Pydantic forward ref resolution
SearchResult.model_rebuild()
