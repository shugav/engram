-- Engram v3 initial schema: Postgres + pgvector + tsvector + JSONB tags.
-- Applied by engram.migrate (schema_migrations table is created by the runner).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'context',
    project TEXT NOT NULL DEFAULT 'default',
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    importance SMALLINT NOT NULL DEFAULT 2
        CHECK (importance >= 0 AND importance <= 4),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'simple',
            coalesce(content, '') || ' ' || coalesce(tags::text, '')
        )
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_memories_project_updated
    ON memories (project, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_project_type
    ON memories (project, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_project_importance
    ON memories (project, importance);
CREATE INDEX IF NOT EXISTS idx_memories_tags_gin ON memories USING GIN (tags jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_memories_content_tsv ON memories USING GIN (content_tsv);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL DEFAULT '',
    embedding vector(1536),
    chunk_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(chunk_text, ''))
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_chunks_memory ON chunks (memory_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks (chunk_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (chunk_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    rel_type TEXT NOT NULL DEFAULT 'relates_to',
    strength DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_rel_pair UNIQUE (source_id, target_id, rel_type)
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships (source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships (target_id);

CREATE TABLE IF NOT EXISTS project_meta (
    project TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (project, key)
);
