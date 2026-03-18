#!/usr/bin/env bash
# Engram launcher -- used by remote Cursor instances over SSH.
# Sources env, activates venv, runs the MCP server over stdio.

set -euo pipefail

ENGRAM_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load API key and project config if present
[ -f "$ENGRAM_DIR/.env" ] && set -a && source "$ENGRAM_DIR/.env" && set +a

export PYTHONPATH="$ENGRAM_DIR/src"
exec "$ENGRAM_DIR/.venv/bin/python" -m engram
