#!/usr/bin/env bash
# setup.sh -- One-time development setup for engram contributors.
#
# Installs git hooks and creates a local patterns file for the pre-commit
# safety check. Run this after cloning:
#
#   ./setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== Engram Development Setup ==="
echo ""

# 1. Point git at the project's hooks directory
git -C "$REPO_ROOT" config core.hooksPath .githooks
echo "[ok] Git hooks installed (.githooks/)"

# 2. Create personal patterns file if it doesn't exist
PATTERNS_FILE="$REPO_ROOT/.githooks/patterns.local"
if [ ! -f "$PATTERNS_FILE" ]; then
    cp "$REPO_ROOT/.githooks/patterns.local.example" "$PATTERNS_FILE"
    echo "[ok] Created .githooks/patterns.local (add your hostnames and paths)"
else
    echo "[ok] .githooks/patterns.local already exists"
fi

# 3. Create .env from example if it doesn't exist
if [ ! -f "$REPO_ROOT/.env" ]; then
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    echo "[ok] Created .env (add your OpenAI API key)"
else
    echo "[ok] .env already exists"
fi

echo ""
echo "Done. Next steps:"
echo "  1. Edit .env with your OPENAI_API_KEY"
echo "  2. Edit .githooks/patterns.local with your personal hostnames/paths"
echo "  3. python3 -m venv .venv && source .venv/bin/activate"
echo "  4. pip install -r requirements.txt"
echo ""
