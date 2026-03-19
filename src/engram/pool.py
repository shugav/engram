"""Global async Postgres pool for Engram v3.

Purpose: One shared ``AsyncConnectionPool`` per process (all projects share one DB).
Role: Opened lazily on first use; closed on shutdown where possible.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger("engram")

_pool: AsyncConnectionPool | None = None
_pool_lock = asyncio.Lock()

DEFAULT_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://engram:engram@localhost:5432/engram",
)


async def get_pool() -> AsyncConnectionPool:
    """Return the process-wide async pool, creating and migrating on first call."""
    global _pool
    async with _pool_lock:
        if _pool is None:
            from psycopg_pool import AsyncConnectionPool

            from .migrate import apply_migrations_async

            dsn = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
            _pool = AsyncConnectionPool(
                conninfo=dsn,
                min_size=1,
                max_size=int(os.environ.get("ENGRAM_POOL_MAX", "20")),
                open=False,
                kwargs={"autocommit": False},
            )
            await _pool.open()
            async with _pool.connection() as conn:
                await apply_migrations_async(conn)
            logger.info("Engram Postgres pool ready (migrations applied)")
        return _pool


async def close_pool() -> None:
    """Close the global pool if it was opened."""
    global _pool
    async with _pool_lock:
        if _pool is not None:
            await _pool.close()
            _pool = None
            logger.info("Engram Postgres pool closed")


def close_pool_sync() -> None:
    """Best-effort synchronous pool shutdown (e.g. ``atexit``)."""
    import asyncio

    global _pool
    if _pool is None:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(close_pool())
    else:
        logger.warning("close_pool_sync skipped: event loop is already running")
