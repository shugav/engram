#!/usr/bin/env bash
# setup-remote.sh -- Configure a remote Cursor instance to use engram over the network.
#
# Run this on any machine to point its Cursor at your engram server.
#
# Usage:
#   bash setup-remote.sh your-server                   # connect to your-server:8788
#   bash setup-remote.sh your-server 9000              # custom host and port

set -euo pipefail

ENGRAM_HOST="${1:?Usage: bash setup-remote.sh <hostname> [port]}"
ENGRAM_PORT="${2:-8788}"
ENGRAM_URL="http://${ENGRAM_HOST}:${ENGRAM_PORT}/sse"
MCP_CONFIG="$HOME/.cursor/mcp.json"

echo "=== Engram Remote Setup ==="
echo "Server: ${ENGRAM_URL}"
echo "Config: ${MCP_CONFIG}"
echo ""

mkdir -p "$(dirname "$MCP_CONFIG")"

if [ -f "$MCP_CONFIG" ]; then
    # Check if jq is available for safe merging
    if command -v jq &> /dev/null; then
        # Merge engram into existing config
        TEMP=$(mktemp)
        jq --arg url "$ENGRAM_URL" \
           '.mcpServers.engram = { "url": $url }' \
           "$MCP_CONFIG" > "$TEMP"
        mv "$TEMP" "$MCP_CONFIG"
        echo "Merged engram into existing mcp.json"
    else
        echo "WARNING: jq not installed. Cannot safely merge into existing mcp.json."
        echo ""
        echo "Add this manually to $MCP_CONFIG under mcpServers:"
        echo ""
        echo "  \"engram\": {"
        echo "    \"url\": \"${ENGRAM_URL}\""
        echo "  }"
        echo ""
        exit 1
    fi
else
    # Create new config
    cat > "$MCP_CONFIG" << MCPEOF
{
  "mcpServers": {
    "engram": {
      "url": "${ENGRAM_URL}"
    }
  }
}
MCPEOF
    echo "Created new mcp.json"
fi

echo ""
echo "Done! Restart Cursor to connect to engram on ${ENGRAM_HOST}."
echo ""
echo "Test the connection:"
echo "  curl -s http://${ENGRAM_HOST}:${ENGRAM_PORT}/sse"
echo ""
