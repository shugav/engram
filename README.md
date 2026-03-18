# Engram

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MCP Server](https://img.shields.io/badge/MCP-server-black)](https://modelcontextprotocol.io/)

Persistent three-layer memory for AI agents, exposed as an MCP server.

Engram replaces flat handover notes with a real memory system backed by SQLite, semantic embeddings, and a typed knowledge graph. It is designed for agents that need context continuity across sessions, machines, and IDEs.

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
  ┌────▼──┐ ┌─▼────┐ ┌▼─────┐ ┌▼──────────┐
  │Chunker│ │Embed │ │Search│ │   DB       │
  │       │ │OpenAI│ │Engine│ │  SQLite    │
  └───────┘ └──────┘ └──────┘ │  + FTS5    │
                               │  + BLOBs   │
                               │  + Graph   │
                               └────────────┘
```

## Quick Start

### 1) Install

```bash
git clone https://github.com/shugav/engram.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3) Add Engram to Cursor (local stdio mode)

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram/.venv/bin/python",
      "args": ["-m", "engram"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ENGRAM_PROJECT": "default",
        "PYTHONPATH": "/path/to/engram/src"
      }
    }
  }
}
```

Replace `/path/to/engram` with your absolute local path.

### 4) Connect from other machines

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

You can also run `setup-remote.sh` to generate this config automatically.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI key used for embeddings |
| `ENGRAM_PROJECT` | `default` | Project namespace (separate DB file per project) |
| `ENGRAM_DIR` | `~/.engram/` | Directory where project databases are stored |
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
composite = (vector * 0.45)
          + (bm25 * 0.25)
          + (recency * 0.15)
          + (graph_connectivity * 0.15)

final_score = composite * importance_multiplier
```

Where:

- **Vector**: semantic similarity against chunk embeddings
- **BM25**: exact keyword relevance via FTS5
- **Recency**: exponential decay based on last access
- **Graph connectivity**: boost for well-connected memories
- **Importance multiplier**: `importance=0` gets strongest boost, `importance=4` the weakest

## Database Layout

Each project gets its own SQLite file at `~/.engram/{project}.db`.

- `memories` -- memory records and metadata
- `memory_fts` -- FTS5 index
- `chunks` -- chunked text, embedding BLOBs, and dedup hashes
- `relationships` -- typed directed graph edges

WAL mode is enabled for better concurrent read behavior.

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
