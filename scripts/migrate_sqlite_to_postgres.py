#!/usr/bin/env python3
"""One-shot SQLite (~/.engram/*.db) to Postgres migration for Engram v3.

Purpose: Move legacy per-file SQLite projects into the unified Postgres database.
Role: Run manually after setting DATABASE_URL; safe to re-run with idempotent inserts.

Importance values are remapped to v3 scale: old 0->4, 1->3, 2->2, 3->1, 4->0.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.types.json import Json

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engram.embeddings import format_vector_literal, from_blob
from engram.migrate import apply_migrations_sync, ensure_database_exists
from engram.util import normalize_project

IMPORTANCE_REMAP = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
DEFAULT_SQLITE_DIR = Path.home() / ".engram"


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def migrate_file(sqlite_path: Path, conn) -> dict[str, int]:
    """Import one SQLite DB file; returns counts."""
    stats = {"memories": 0, "chunks": 0, "relationships": 0, "meta": 0}
    sq = sqlite3.connect(str(sqlite_path))
    sq.row_factory = sqlite3.Row

    row0 = sq.execute("SELECT project FROM memories LIMIT 1").fetchone()
    if row0 and row0["project"]:
        project = normalize_project(row0["project"])
    else:
        project = normalize_project(sqlite_path.stem)

    with conn.cursor() as cur:
        for row in sq.execute("SELECT * FROM memories"):
            imp = int(row["importance"])
            imp = IMPORTANCE_REMAP.get(imp, imp)
            tags = row["tags"]
            if isinstance(tags, str):
                try:
                    tags_list = json.loads(tags)
                except json.JSONDecodeError:
                    tags_list = []
            else:
                tags_list = list(tags) if tags else []
            cur.execute(
                """INSERT INTO memories (id, content, memory_type, project, tags,
                   importance, access_count, last_accessed, created_at, updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO UPDATE SET
                   content=EXCLUDED.content, updated_at=EXCLUDED.updated_at""",
                (
                    row["id"],
                    row["content"],
                    row["memory_type"],
                    project,
                    Json(tags_list),
                    imp,
                    row["access_count"],
                    _parse_ts(row["last_accessed"]),
                    _parse_ts(row["created_at"]),
                    _parse_ts(row["updated_at"]),
                ),
            )
            stats["memories"] += 1

        for row in sq.execute("SELECT * FROM chunks"):
            emb = row["embedding"]
            if emb:
                lit = format_vector_literal(from_blob(bytes(emb)))
                cur.execute(
                    """INSERT INTO chunks (id, memory_id, chunk_text, chunk_index, chunk_hash, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s::vector)
                       ON CONFLICT (id) DO NOTHING""",
                    (
                        row["id"],
                        row["memory_id"],
                        row["chunk_text"],
                        row["chunk_index"],
                        row["chunk_hash"] or "",
                        lit,
                    ),
                )
            else:
                cur.execute(
                    """INSERT INTO chunks (id, memory_id, chunk_text, chunk_index, chunk_hash, embedding)
                       VALUES (%s,%s,%s,%s,%s,NULL)
                       ON CONFLICT (id) DO NOTHING""",
                    (
                        row["id"],
                        row["memory_id"],
                        row["chunk_text"],
                        row["chunk_index"],
                        row["chunk_hash"] or "",
                    ),
                )
            stats["chunks"] += 1

        for row in sq.execute("SELECT * FROM relationships"):
            cur.execute(
                """INSERT INTO relationships (id, source_id, target_id, rel_type, strength, created_at)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (source_id, target_id, rel_type) DO NOTHING""",
                (
                    row["id"],
                    row["source_id"],
                    row["target_id"],
                    row["rel_type"],
                    float(row["strength"]),
                    _parse_ts(row["created_at"]),
                ),
            )
            stats["relationships"] += 1

        try:
            for row in sq.execute("SELECT key, value FROM project_meta"):
                cur.execute(
                    """INSERT INTO project_meta (project, key, value)
                       VALUES (%s,%s,%s)
                       ON CONFLICT (project, key) DO UPDATE SET value=EXCLUDED.value""",
                    (project, row["key"], row["value"]),
                )
                stats["meta"] += 1
        except sqlite3.OperationalError:
            pass

    sq.close()
    return stats


def main() -> int:
    dsn = os.environ.get("DATABASE_URL", "postgresql://engram:engram@localhost:5432/engram")
    src_dir = Path(os.environ.get("ENGRAM_SQLITE_DIR", str(DEFAULT_SQLITE_DIR)))

    ensure_database_exists(dsn)
    files = sorted(src_dir.glob("*.db"))
    if not files:
        print(f"No *.db files in {src_dir}")
        return 0

    with psycopg.connect(dsn) as conn:
        apply_migrations_sync(conn)
        total = {"memories": 0, "chunks": 0, "relationships": 0, "meta": 0}
        for f in files:
            print(f"Migrating {f.name} ...", flush=True)
            st = migrate_file(f, conn)
            for k in total:
                total[k] += st[k]
            print(f"  {st}", flush=True)
        conn.commit()
    print("Done:", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
