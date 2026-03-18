"""Engram MCP Server -- persistent three-layer memory for AI agents."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from .db import MemoryDB
from .embeddings import create_embedder
from .errors import EmbeddingConfigMismatchError
from .search import SearchEngine
from .types import (
    Memory,
    MemoryType,
    Relationship,
    RelationType,
)

ENGRAM_INSTRUCTIONS = """\
You have access to Engram, a persistent memory system shared across all of the \
user's machines. Memories survive across sessions, workspaces, and devices. \
Other AI agents working for this user also read and write to this same system.

## CRITICAL -- Project Scoping

Every tool accepts a `project` parameter. You MUST set this to the current \
workspace/project name (lowercase, hyphenated). Derive it from the workspace \
folder name -- for example, if you are working in `/home/user/my-cool-app`, \
set project="my-cool-app".

Each project gets its own isolated database. Memories stored in project "my-app" \
are invisible to project "web-dashboard" and vice versa. This prevents cross-project \
pollution.

For user-wide preferences that should apply everywhere (e.g. "user prefers dark mode", \
"always use Tailscale hostnames"), store them in project="global" so any project \
can find them.

## CRITICAL -- Session Start (Recall Before Working)

At the START of every task, before writing any code:

1. Call memory_recall with query "session handoff" AND the current project name \
   to find where the last agent left off.
2. Call memory_recall with a task-relevant query AND the current project name.
3. Also call memory_recall with the same query against project="global" to pick \
   up user-wide preferences.

If a session handoff note is found, present it to the user first and ask if they \
want to continue from there.

## CRITICAL -- Session End (Handoff Before Finishing)

Before your FINAL response in any significant task, you MUST store a session \
handoff memory:

memory_store(content="SESSION HANDOFF: [what was done] | NEXT: [what should \
happen next] | BLOCKED: [blockers] | FILES CHANGED: [files modified]", \
memory_type="context", tags="session-handoff", importance=1, project="<project>")

This is how the next agent picks up exactly where you left off. Think of it like \
a nurse handing off to the next shift. Every significant task ends with a handoff.

## When to Store Memories

Store a memory whenever you encounter something a future agent would benefit \
from knowing:

- **Decisions** (type: decision): "Chose PostgreSQL over MySQL because ..."
- **Patterns** (type: pattern): "This codebase uses repository pattern for DB access"
- **Errors** (type: error): "Port 3000 is already in use on my-server"
- **Architecture** (type: architecture): "Auth flow: JWT -> middleware -> httpOnly cookie"
- **Preferences** (type: preference): "User prefers tabs over spaces" \
  (store in project="global" if user-wide)
- **Context** (type: context): General project/environment details

## Importance Levels

- 0 = Critical identity (core user preferences, system-wide decisions)
- 1 = Key facts (important project decisions, recurring patterns)
- 2 = General context (default -- most memories go here)
- 3 = Low priority (minor notes, temporary context)
- 4 = Trivial (auto-pruned after 30 days if never accessed)

## Tags

Always add relevant tags. Use short, lowercase, hyphenated tags: \
"auth", "docker", "tailscale", "python", "frontend".

## Knowledge Graph

After storing related memories, use memory_connect to link them. Connected \
memories surface automatically during recall.

## Feedback Loop

After using recall results, call memory_feedback to mark them helpful or not. \
This trains the graph to surface better results over time.

## Correcting Wrong Memories

When you discover a stored memory is wrong or outdated, use memory_correct:

memory_correct(old_memory_id="<id>", new_content="The corrected info", \
project="<project>")

This stores the correction, links it to the old memory via a "supersedes" \
relationship, and demotes the old memory so it gets pruned over time. \
The old memory is NOT deleted -- the history is preserved.

If a recalled memory has a WARNING field saying "THIS MEMORY HAS BEEN \
SUPERSEDED", always prefer the newer version referenced in "superseded_by".

Correct proactively: if you discover during your work that a stored memory \
is wrong (file path changed, dependency updated, approach abandoned), \
fix it immediately without being asked.

## Maintenance

Run memory_consolidate periodically to deduplicate, decay unused edges, and \
prune stale memories.

## Onboarding New Projects

When you first connect to a project with zero memories, store foundational \
context: what the project is, its tech stack, key architecture decisions, \
and the user's conventions for this codebase. This bootstraps future agents.
"""

mcp = FastMCP("engram", instructions=ENGRAM_INSTRUCTIONS)

_engines: dict[str, SearchEngine] = {}


def _get_engine(project: str | None = None) -> SearchEngine:
    """Return (or create) a SearchEngine for the given project.

    Each project gets its own SQLite database file (~/.engram/{project}.db),
    so memories are fully isolated between projects.
    """
    project = (project or os.environ.get("ENGRAM_PROJECT", "default")).strip().lower()
    if project not in _engines:
        db_dir = os.environ.get("ENGRAM_DIR", None)
        db = MemoryDB(project=project, db_dir=db_dir)
        embedder = create_embedder()
        _engines[project] = SearchEngine(db=db, embedder=embedder)
    return _engines[project]


@mcp.tool()
def memory_store(
    content: str,
    memory_type: str = "context",
    tags: str = "",
    importance: int = 2,
    project: str = "",
) -> dict:
    """Store a new memory. Auto-chunks, embeds, and indexes for three-layer search.

    Args:
        content: The memory content to store. Be specific and detailed.
        memory_type: One of: decision, pattern, error, context, architecture, preference.
        tags: Comma-separated tags for filtering (e.g. "auth,security,jwt").
        importance: Priority 0-4. 0=critical identity, 1=key facts, 2=general, 3=low, 4=trivial.
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        The stored memory's ID and metadata.
    """
    engine = _get_engine(project or None)

    try:
        mt = MemoryType(memory_type)
    except ValueError:
        mt = MemoryType.CONTEXT

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    importance = max(0, min(4, importance))

    memory = Memory(
        content=content,
        memory_type=mt,
        tags=tag_list,
        importance=importance,
    )

    try:
        stored = engine.store(memory)
    except EmbeddingConfigMismatchError as e:
        return {"error": str(e)}

    return {
        "status": "stored",
        "id": stored.id,
        "memory_type": stored.memory_type.value,
        "tags": stored.tags,
        "importance": stored.importance,
    }


@mcp.tool()
def memory_recall(
    query: str,
    top_k: int = 5,
    memory_type: str = "",
    tags: str = "",
    min_importance: int = 4,
    graph_hops: int = 1,
    project: str = "",
) -> dict:
    """Search memories using all three layers: keyword (BM25), semantic (vector), and graph.

    Results are ranked by a composite score:
      Final = (vector * 0.45 + BM25 * 0.25 + recency * 0.15 + graph * 0.15) * importance_multiplier

    Connected memories from the knowledge graph are attached automatically.

    Args:
        query: What to search for. Can be a keyword, question, or concept.
        top_k: Number of results to return (default 5).
        memory_type: Filter by type (decision/pattern/error/context/architecture/
            preference). Empty = all.
        tags: Comma-separated tags to filter by. Empty = all.
        min_importance: Only return memories with importance <= this value (0=only critical, 4=all).
        graph_hops: How many relationship hops to traverse (1 or 2).
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Ranked list of memories with scores, matched chunks, and connected context.
    """
    engine = _get_engine(project or None)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    mt = memory_type if memory_type else None
    mi = min_importance if min_importance < 4 else None

    results = engine.recall(
        query=query,
        top_k=top_k,
        memory_type=mt,
        tags=tag_list,
        min_importance=mi,
        graph_hops=max(1, min(2, graph_hops)),
    )

    output = []
    for r in results:
        # Check if this memory has been superseded by a newer one
        superseded_by = None
        for c in r.connected:
            if c.rel_type == "supersedes" and c.direction == "incoming":
                superseded_by = {
                    "id": c.memory.id,
                    "content": c.memory.content[:300],
                }
                break

        entry = {
            "id": r.memory.id,
            "content": r.memory.content,
            "type": r.memory.memory_type.value,
            "tags": r.memory.tags,
            "importance": r.memory.importance,
            "score": r.score,
            "score_breakdown": r.score_breakdown,
            "matched_chunk": r.matched_chunk,
            "connected": [
                {
                    "id": c.memory.id,
                    "content": c.memory.content[:300],
                    "rel_type": c.rel_type,
                    "direction": c.direction,
                    "strength": c.strength,
                }
                for c in r.connected
            ],
        }

        if superseded_by:
            entry["WARNING"] = "THIS MEMORY HAS BEEN SUPERSEDED. Use the newer version instead."
            entry["superseded_by"] = superseded_by

        output.append(entry)

    return {"results": output, "count": len(output)}


@mcp.tool()
def memory_connect(
    source_id: str,
    target_id: str,
    rel_type: str = "relates_to",
    strength: float = 1.0,
    project: str = "",
) -> dict:
    """Create a typed relationship between two memories in the knowledge graph.

    This is how memories become interconnected. When one memory is recalled,
    its connected memories are pulled in automatically.

    Args:
        source_id: ID of the source memory.
        target_id: ID of the target memory.
        rel_type: Type: caused_by, relates_to, depends_on,
            supersedes, used_in, resolved_by.
        strength: Connection strength from 0.0 to 1.0 (default 1.0).
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        The created relationship.
    """
    engine = _get_engine(project or None)

    source = engine.db.get_memory(source_id)
    target = engine.db.get_memory(target_id)
    if not source:
        return {"error": f"Source memory '{source_id}' not found."}
    if not target:
        return {"error": f"Target memory '{target_id}' not found."}

    try:
        rt = RelationType(rel_type)
    except ValueError:
        rt = RelationType.RELATES_TO

    rel = Relationship(
        source_id=source_id,
        target_id=target_id,
        rel_type=rt,
        strength=max(0.0, min(1.0, strength)),
    )
    engine.db.store_relationship(rel)

    return {
        "status": "connected",
        "id": rel.id,
        "source_id": source_id,
        "target_id": target_id,
        "rel_type": rt.value,
        "strength": rel.strength,
    }


@mcp.tool()
def memory_list(
    memory_type: str = "",
    tags: str = "",
    min_importance: int = 4,
    limit: int = 20,
    project: str = "",
) -> dict:
    """List recent memories with optional filters.

    Args:
        memory_type: Filter by type. Empty = all types.
        tags: Comma-separated tags to filter by. Empty = all.
        min_importance: Only return memories with importance <= this (0=only critical, 4=all).
        limit: Max number of memories to return.
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        List of memories sorted by most recently updated.
    """
    engine = _get_engine(project or None)

    mt = MemoryType(memory_type) if memory_type else None
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    mi = min_importance if min_importance < 4 else None

    memories = engine.db.list_memories(
        memory_type=mt,
        tags=tag_list,
        min_importance=mi,
        limit=limit,
    )

    return {
        "memories": [
            {
                "id": m.id,
                "content": m.content[:300],
                "type": m.memory_type.value,
                "tags": m.tags,
                "importance": m.importance,
                "access_count": m.access_count,
                "created_at": m.created_at.isoformat(),
                "updated_at": m.updated_at.isoformat(),
            }
            for m in memories
        ],
        "count": len(memories),
    }


@mcp.tool()
def memory_correct(
    old_memory_id: str,
    new_content: str,
    memory_type: str = "",
    tags: str = "",
    importance: int = 1,
    project: str = "",
) -> dict:
    """Correct or supersede a wrong/outdated memory.

    Use this when a recalled memory contains wrong information, an outdated
    decision, a bug fix that turned out to be incorrect, or anything that
    should no longer be trusted. The old memory is demoted and linked to the
    new one via a 'supersedes' relationship. Future recalls will prefer the
    new memory and deprioritize the old one.

    Args:
        old_memory_id: ID of the memory that is wrong or outdated.
        new_content: The corrected/updated information.
        memory_type: Type for the new memory. Empty = inherit from old memory.
        tags: Comma-separated tags. Empty = inherit from old memory.
        importance: Importance for the new memory (default 1 = high).
        project: Project namespace. Derive from workspace folder name. Empty = "default".

    Returns:
        The new memory ID and confirmation that the old one was superseded.
    """
    engine = _get_engine(project or None)

    old_mem = engine.db.get_memory(old_memory_id)
    if not old_mem:
        return {"error": f"Memory '{old_memory_id}' not found."}

    if not memory_type:
        mt = old_mem.memory_type
    else:
        try:
            mt = MemoryType(memory_type)
        except ValueError:
            mt = old_mem.memory_type

    tag_list = (
        [t.strip() for t in tags.split(",") if t.strip()] if tags else old_mem.tags
    )

    new_memory = Memory(
        content=new_content,
        memory_type=mt,
        tags=tag_list,
        importance=max(0, min(4, importance)),
    )
    try:
        stored = engine.store(new_memory)
    except EmbeddingConfigMismatchError as e:
        return {"error": str(e)}

    # Link: new supersedes old
    rel = Relationship(
        source_id=stored.id,
        target_id=old_memory_id,
        rel_type=RelationType.SUPERSEDES,
        strength=1.0,
    )
    engine.db.store_relationship(rel)

    # Demote old memory to trivial so it gets pruned over time
    engine.db.update_memory(old_memory_id, importance=4)

    return {
        "status": "corrected",
        "old_id": old_memory_id,
        "old_content_preview": old_mem.content[:200],
        "new_id": stored.id,
        "new_content_preview": new_content[:200],
        "relationship": "new supersedes old",
        "old_demoted_to": "trivial (will be pruned if unused)",
    }


@mcp.tool()
def memory_forget(memory_id: str, project: str = "") -> dict:
    """Remove a memory and all its relationships from the knowledge graph.

    Args:
        memory_id: The ID of the memory to remove.
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Confirmation of deletion.
    """
    engine = _get_engine(project or None)

    mem = engine.db.get_memory(memory_id)
    if not mem:
        return {"error": f"Memory '{memory_id}' not found."}

    engine.db.delete_chunks_for_memory(memory_id)
    engine.db.delete_relationships_for_memory(memory_id)
    engine.db.delete_memory(memory_id)

    return {"status": "forgotten", "id": memory_id}


@mcp.tool()
def memory_status(project: str = "") -> dict:
    """Get statistics about the memory system.

    Args:
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Total memories, chunks, relationships, breakdown by type and importance,
        database size, and age range.
    """
    engine = _get_engine(project or None)
    stats = engine.db.get_stats()
    return stats.model_dump()


@mcp.tool()
def memory_feedback(
    memory_ids: str,
    helpful: bool = True,
    project: str = "",
) -> dict:
    """Provide feedback on recall results to strengthen or weaken graph connections.

    When recall results are helpful, their graph edges get reinforced -- making
    those connections more likely to surface in future recalls. When unhelpful,
    edges weaken. Over time the knowledge graph self-optimizes based on what
    actually helps you.

    Call this after memory_recall when you know whether the results were useful.

    Args:
        memory_ids: Comma-separated IDs of memories from the recall results.
        helpful: True if the results were useful, False if they were not.
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Number of memories whose graph edges were adjusted.
    """
    engine = _get_engine(project or None)
    ids = [mid.strip() for mid in memory_ids.split(",") if mid.strip()]
    if not ids:
        return {"error": "No memory IDs provided."}
    result = engine.feedback(ids, helpful)
    return result


@mcp.tool()
def memory_consolidate(project: str = "") -> dict:
    """Run a memory enhancement pass (inspired by Cognee's memify).

    Three stages:
    1. Deduplicates chunks by hash to remove exact duplicates.
    2. Applies temporal decay to all graph edges and prunes weak connections
       (strength < 0.1) -- edges that are never reinforced by feedback fade away.
    3. Prunes stale, never-accessed, low-importance memories older than 30 days.

    Frequently-used connections survive and strengthen. Unused ones decay.
    Run this periodically to keep the memory system healthy and focused.

    Args:
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Breakdown of chunks deduped, edges decayed/pruned, and stale memories removed.
    """
    engine = _get_engine(project or None)
    result = engine.memify()
    return {"status": "consolidated", **result}


@mcp.prompt()
def onboarding(project: str = "") -> str:
    """Get a quick-start guide for using engram effectively. Call this if you're
    unsure how to use the memory system or want a refresher on best practices.

    Args:
        project: Project namespace to show stats for. Empty = "default".
    """
    proj = (project or "default").strip().lower()
    engine = _get_engine(proj)
    stats = engine.db.get_stats()
    s = stats.model_dump()

    mem_count = s.get("total_memories", 0)
    is_new = mem_count == 0

    header = (
        f"# Engram Quick-Start -- project: `{proj}`\n\n"
        f"**Memory DB status:** {mem_count} memories, "
        f"{s.get('total_chunks', 0)} chunks, "
        f"{s.get('total_relationships', 0)} graph edges.\n\n"
    )

    bootstrap = ""
    if is_new:
        bootstrap = (
            "## NEW PROJECT -- Bootstrap Required\n\n"
            "This project has zero memories. You should store foundational context:\n\n"
            "1. **What is this project?** Purpose, goals, current status.\n"
            "2. **Tech stack:** Languages, frameworks, databases, infra.\n"
            "3. **Architecture:** Key patterns, data flow, directory structure.\n"
            "4. **Conventions:** Coding style, naming, testing approach.\n"
            "5. **Known issues:** Current bugs, tech debt, gotchas.\n\n"
            "Use type `architecture` for #2-3, type `context` for #1, "
            "type `preference` for #4, type `error` for #5.\n\n"
            "Also recall from project=`global` for user-wide preferences.\n\n"
        )

    workflow = (
        "## Your Workflow\n\n"
        f"1. **Recall first:** `memory_recall('topic', project='{proj}')`\n"
        f"2. **Also check global:** `memory_recall('topic', project='global')`\n"
        "3. **Work:** Use recalled context to inform your decisions.\n"
        f"4. **Store:** `memory_store('...', project='{proj}')`\n"
        f"5. **Connect:** `memory_connect(src, tgt, project='{proj}')`\n"
        "6. **Feedback:** Mark recall results helpful/unhelpful.\n\n"
    )

    types_and_tips = (
        "## Memory Types\n\n"
        "| Type | Use for |\n"
        "|------|--------|\n"
        "| decision | Choices made and their reasoning |\n"
        "| pattern | Recurring code/architecture patterns |\n"
        "| error | Bugs, gotchas, and their fixes |\n"
        "| architecture | System design, data flow, integrations |\n"
        "| preference | User preferences and conventions |\n"
        "| context | General project/environment context |\n\n"
        "## Project Scoping\n\n"
        f"- **This project:** `project='{proj}'` -- for project-specific memories.\n"
        "- **User-wide:** `project='global'` -- for preferences that apply everywhere.\n"
        "- Never mix: don't store project-specific decisions in global, or vice versa.\n\n"
        "## Tips\n\n"
        "- Be specific. 'Auth uses JWT' < 'Auth uses RS256 JWT issued by /api/login "
        "with 24h expiry in httpOnly cookie.'\n"
        "- Always add tags. Future recall depends on them.\n"
        "- Use importance 0-1 sparingly -- only for things that should never be pruned.\n"
    )

    return header + bootstrap + workflow + types_and_tips


def _wrap_with_api_key_auth(app, api_key: str):
    """ASGI middleware that rejects requests missing a valid Bearer token.

    Uses constant-time comparison to prevent timing side-channel attacks.
    """
    import secrets

    from starlette.responses import JSONResponse

    expected = f"Bearer {api_key}".encode("utf-8")

    async def auth_middleware(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            token = headers.get(b"authorization", b"")
            if not secrets.compare_digest(token, expected):
                resp = JSONResponse({"error": "unauthorized"}, status_code=401)
                await resp(scope, receive, send)
                return
        await app(scope, receive, send)

    return auth_middleware


def main(
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 8788,
    api_key: str | None = None,
) -> None:
    """Start the engram MCP server.

    Args:
        transport: "stdio" for local subprocess, "sse" for network HTTP/SSE.
        host: Bind address for SSE mode.
        port: Port for SSE mode.
        api_key: Optional Bearer token for SSE auth.
    """
    if transport == "stdio":
        mcp.run()
    elif transport == "sse":
        import anyio
        import uvicorn

        # Disable DNS rebinding protection for network access
        mcp.settings.transport_security = None

        app = mcp.sse_app()

        if api_key:
            app = _wrap_with_api_key_auth(app, api_key)

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        anyio.run(server.serve)


if __name__ == "__main__":
    main()
