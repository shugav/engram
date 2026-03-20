# Engram

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MCP Server](https://img.shields.io/badge/MCP-server-black)](https://modelcontextprotocol.io/)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange)](https://github.com/shugav/engram)

> **Beta software.** Engram is under active development. APIs, storage format,
> and behavior may change between releases. Use in production at your own
> discretion. See [LICENSE](LICENSE) for the full warranty disclaimer.

Persistent three-layer memory for AI agents, exposed as an MCP server.

**v3** uses **Postgres + pgvector** (async `psycopg`), `tsvector` keyword search, JSONB tags, and a typed knowledge graph. Docker Compose is the recommended install. Legacy SQLite data can be migrated with `scripts/migrate_sqlite_to_postgres.py`.

## Why Engram

Most agents forget everything between sessions. Plain text notes help, but they are hard to search, hard to connect, and easy to let rot.

Engram gives agents a durable "second brain" with:

- **BM25 keyword recall** for exact terms like error codes, IDs, and symbol names
- **Vector semantic recall** for related ideas even when wording differs
- **Knowledge graph expansion** so connected memories appear together
- **Recency + importance ranking** so critical and recent memories float to the top
- **Feedback loops** so recall quality improves over time

## Architecture

```text
┌─────────────────────────────────────────────────┐
│        AI Agent (Cursor, VS Code, Claude, ...) │
└──────────────┬──────────────────────────────────┘
               │ MCP (stdio or SSE)
┌──────────────▼──────────────────────────────────┐
│              Engram MCP Server                  │
│  Tools: store, recall, connect, list, correct, │
│  forget, status, feedback, consolidate          │
└──────┬──────┬───────┬───────┬───────────────────┘
       │      │       │       │
  ┌────▼──┐ ┌─▼────┐ ┌▼─────┐ ┌▼──────────────┐
  │Chunker│ │Embed │ │Search│ │ Postgres      │
  │       │ │async │ │Engine│ │ + pgvector    │
  └───────┘ └──┬───┘ └──────┘ │ + tsvector    │
          ┌────┼────┐         │ + JSONB tags  │
          │ OpenAI  │         │ + graph edges │
          │ Ollama  │         └───────────────┘
          │  None   │
          └─────────┘
```

## Quick Start

### 1) Docker (recommended)

```bash
git clone https://github.com/shugav/engram.git
cd engram
docker compose up --build
```

Set `OPENAI_API_KEY` / `ENGRAM_EMBEDDER` / `ENGRAM_API_KEY` in the environment or a `.env` file as needed. Postgres listens on `localhost:5432`, Engram SSE on `8788`.

### 2) Local install (dev)

```bash
git clone https://github.com/shugav/engram.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export DATABASE_URL="postgresql://user:pass@localhost:5432/engram"
```

Run a **pgvector** image (e.g. `pgvector/pgvector:pg16`), create a database, then start Engram. Migrations apply automatically on first pool open.

### 3) SQLite → Postgres migration

If you have legacy `~/.engram/*.db` files from v2:

```bash
export DATABASE_URL="postgresql://..."
python scripts/migrate_sqlite_to_postgres.py
```

Optional: `ENGRAM_SQLITE_DIR=/path/to/dbs` to override the source directory.

### Tests (developers)

Requires Postgres with **pgvector** (same image as production). The suite resets the `public` schema once per session.

```bash
export ENGRAM_TEST_DATABASE_URL="postgresql://engram:engram@127.0.0.1:5432/engram_test"
pytest
```

### 4) Configure embeddings (optional)

Engram works in three embedding modes. It auto-detects the best available:

| Mode | Quality | Setup | Env var |
|------|---------|-------|---------|
| **OpenAI** | Highest | `export OPENAI_API_KEY="sk-..."` | `ENGRAM_EMBEDDER=openai` |
| **Ollama** | Good (free, local) | [Install Ollama](https://ollama.com), then `ollama pull nomic-embed-text` | `ENGRAM_EMBEDDER=ollama` |
| **None** | BM25 keyword only | Nothing needed | `ENGRAM_EMBEDDER=none` |

If no embedder is configured, engram auto-detects: Ollama (if running) -> OpenAI (if key set) -> BM25-only.

### 5) Add Engram to Cursor (local stdio mode)

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram/.venv/bin/python",
      "args": ["-m", "engram"],
      "env": {
        "OPENAI_API_KEY": "<your-openai-key>",
        "DATABASE_URL": "postgresql://engram:engram@localhost:5432/engram",
        "ENGRAM_PROJECT": "default",
        "PYTHONPATH": "/path/to/engram/src"
      }
    }
  }
}
```

Replace `/path/to/engram` with your absolute local path.

### 6) Connect from other machines

#### Option A: SSH + stdio (simple and reliable)

On the remote machine's `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "ssh",
      "args": ["your-server", "/path/to/engram/engram.sh"],
      "env": {}
    }
  }
}
```

#### Option B: SSE server mode (network endpoint)

Start server:

```bash
python -m engram --transport sse --host 0.0.0.0 --port 8788
```

Then configure clients with:

```json
{
  "mcpServers": {
    "engram": {
      "url": "http://your-server:8788/sse"
    }
  }
}
```

You can also run `setup-remote.sh` to generate this config automatically:

```bash
# Run on any machine to point Cursor at your engram server
bash setup-remote.sh your-server          # uses default port 8788
bash setup-remote.sh your-server 9000     # custom port
```

### 7) Multi-machine rule sync

**Source of Truth:** The canonical version lives in the `global` project as a memory tagged `engram-rule-sync` (the memory with the highest `RULE_VERSION` is authoritative). The local `~/.cursor/rules/engram-memory.mdc` is a cached copy that agents should sync from the server on session start. **Never edit the local rule directly** -- update the canonical memory instead (using `memory_correct` with bumped version).

When running Engram across multiple machines (e.g. SSE over Tailscale), cursor rules auto-sync through Engram itself.

**How it works:** A canonical copy of the Engram cursor rule is stored as a memory in the `global` project with the tag `engram-rule-sync`. Each machine's rule file includes a version check at session start. If the server has a newer version, the agent silently overwrites the local file.

**First-time setup on a new machine** -- paste this into any Cursor chat:

```
Read the Engram MCP memory using: memory_list(memory_type="pattern", tags="engram-rule-sync", project="global").
Extract everything below the "---" separator and write it to ~/.cursor/rules/engram-memory.mdc.
```

**Pushing rule updates:** Edit the rule on your primary machine, bump the `RULE_VERSION` comment at the top, then store the new version:

```
memory_correct(old_memory_id="<id of current engram-rule-sync memory>",
               new_content="ENGRAM_RULE_CANONICAL RULE_VERSION:X.Y\n---\n<full rule content>",
               project="global")
```

All other machines pick up the change on their next session start.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENGRAM_EMBEDDER` | auto-detect | Embedding provider: `openai`, `ollama`, or `none` |
| `OPENAI_API_KEY` | (unset) | OpenAI key (only needed if using OpenAI embeddings) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL (only needed if non-default) |
| `ENGRAM_PROJECT` | `default` | Project namespace (column-level isolation) |
| `DATABASE_URL` | `postgresql://engram:engram@localhost:5432/engram` | Postgres connection string |
| `ENGRAM_API_KEY` | (unset) | Optional bearer token required for SSE requests |

## MCP Tools

### Memory operations

- `memory_store` -- add a memory with type, tags, and importance
- `memory_recall` -- hybrid search (BM25 + vector + graph + recency)
- `memory_list` -- list recent memories with optional filters
- `memory_correct` -- supersede outdated or wrong memory with corrected content
- `memory_forget` -- delete a memory and its relationships

### Graph and quality operations

- `memory_connect` -- create typed relations (`depends_on`, `supersedes`, etc.)
- `memory_feedback` -- reinforce or weaken graph edges based on recall quality
- `memory_consolidate` -- dedup chunks, decay weak edges, prune stale low-value memories
- `memory_status` -- view stats (memories, chunks, graph edges, DB size)

### Prompt

- `onboarding` -- returns a usage guide tuned to the selected project

## Ranking Model

Search results are scored with this composite function:

```text
# With embeddings (default)
composite = (vector * 0.45) + (bm25 * 0.25) + (recency * 0.15) + (graph * 0.15)

# Null / BM25-only embedder: vector weight is redistributed (e.g. bm25 0.50, recency 0.30, graph 0.20)

final_score = composite * importance_multiplier   # mult ≈ 1.0 + importance * 0.125 (v3)
```

Where:

- **Vector**: semantic similarity against chunk embeddings
- **BM25**: exact keyword relevance via Postgres `tsvector` / `ts_rank_cd`
- **Recency**: exponential decay based on last access
- **Graph connectivity**: boost for well-connected memories
- **Importance multiplier** (v3): higher `importance` (up to 4) increases the multiplier

## Database layout (Postgres)

One database, many projects: the `project` column isolates tenants.

- `memories` -- content, JSONB tags, `tsvector` for search, importance, timestamps
- `chunks` -- chunked text, `vector(1536)` embeddings (padded), dedup hashes
- `relationships` -- typed directed graph edges (FK + `ON DELETE CASCADE`)
- `project_meta` -- embedding provider metadata per `(project, key)`

## Architecture & Scaling

### Deployment modes

| Mode | Agents | How |
|------|--------|-----|
| **stdio** (local) | Single agent per machine | Cursor spawns engram as a subprocess |
| **SSE** (network) | Many agents, one server | Run one central server; agents connect as clients |

For multiple concurrent agents, **use SSE mode** with **one** Engram container (or process) and a shared Postgres. Clients talk MCP over HTTP; the server uses a connection pool.

### Known limitations

- **Embedding model lock-in per project**: After the first vector write, switching embedder name fails fast (`EmbeddingConfigMismatchError`). Re-embed or use a fresh project namespace.
- **Fixed vector width**: Chunks use `vector(1536)`; shorter provider outputs are zero-padded.

## Compatible MCP Clients

Engram works with any MCP-compatible client, including:

- Cursor
- VS Code (Copilot MCP support)
- Claude Desktop
- Windsurf
- Claude Code

## Community and Support

- Contributing guide: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Support guide: `SUPPORT.md`
- Security policy: `SECURITY.md`

## Contributing

Contributions are welcome. First-time contributors are encouraged.
See `CONTRIBUTING.md` for setup and workflow.

## License

MIT License. See `LICENSE`.
