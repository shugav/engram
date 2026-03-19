"""Shared helpers used across server, database, and migration layers.

Purpose: Single source of truth for project name normalization so the in-process
engine cache key always matches the on-disk database filename.
"""

from __future__ import annotations

import os
import re


def normalize_project(name: str | None) -> str:
    """Normalize a project name for the Postgres ``project`` column and engine cache.

    Steps: strip, lower, then keep only [a-zA-Z0-9_-]. Empty result becomes
    ``default``. Matches ENGRAM_PROJECT env when ``name`` is empty/None.

    Args:
        name: Raw project string from MCP ``project`` parameter, or None.

    Returns:
        Sanitized project string used for multi-tenant isolation.
    """
    raw = (name if name is not None else "") or os.environ.get("ENGRAM_PROJECT", "default")
    raw = str(raw).strip().lower()
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", raw) or "default"
    return sanitized
