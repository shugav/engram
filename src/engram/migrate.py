"""Apply SQL migrations to Postgres (Engram v3).

Purpose: Versioned schema upgrades. Role: Invoked when opening the connection pool.
"""

from __future__ import annotations

import re
from pathlib import Path

from psycopg import sql
from psycopg.conninfo import conninfo_to_dict


def migrations_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "migrations"


def _split_sql_statements(sql_text: str) -> list[str]:
    """Split a migration file into executable statements (skips empty / comment-only)."""
    statements: list[str] = []
    buf: list[str] = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(buf).strip()
            buf = []
            if stmt:
                statements.append(stmt)
    if buf:
        tail = "\n".join(buf).strip()
        if tail:
            statements.append(tail)
    return statements


async def apply_migrations_async(conn) -> None:
    """Run pending migrations. ``conn`` is an async psycopg connection."""
    async with conn.cursor() as cur:
        await cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    await conn.commit()

    files = sorted(migrations_dir().glob("*.sql"))
    for path in files:
        m = re.match(r"^(\d+)_", path.name)
        if not m:
            continue
        version = int(m.group(1))
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s",
                (version,),
            )
            if await cur.fetchone():
                continue

        sql_text = path.read_text(encoding="utf-8")
        for stmt in _split_sql_statements(sql_text):
            async with conn.cursor() as cur:
                await cur.execute(stmt)
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
                (version,),
            )
        await conn.commit()


def apply_migrations_sync(conn) -> None:
    """Sync variant for tooling/tests that use a blocking connection."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    conn.commit()

    files = sorted(migrations_dir().glob("*.sql"))
    for path in files:
        m = re.match(r"^(\d+)_", path.name)
        if not m:
            continue
        version = int(m.group(1))
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s",
                (version,),
            )
            if cur.fetchone():
                continue

        sql_text = path.read_text(encoding="utf-8")
        for stmt in _split_sql_statements(sql_text):
            with conn.cursor() as cur:
                cur.execute(stmt)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
                (version,),
            )
        conn.commit()


def ensure_database_exists(dsn: str) -> None:
    """Create the target database if missing (requires CREATEDB on the role)."""
    info = conninfo_to_dict(dsn)
    dbname = info.get("dbname") or "postgres"
    info_admin = {**info, "dbname": "postgres"}
    try:
        import psycopg

        with psycopg.connect(**info_admin, autocommit=True) as admin:
            with admin.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (dbname,),
                )
                if cur.fetchone():
                    return
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    except Exception:
        pass
