"""Entry point for `python -m engram`.

Supports two transport modes:
  - stdio (default): Local subprocess transport for Cursor on the same machine.
  - sse:  HTTP/SSE transport for remote Cursor instances over the network.

Examples:
    python -m engram                        # local stdio (default)
    python -m engram --transport sse        # network SSE on 0.0.0.0:8788
    python -m engram --transport sse --port 9000
"""

import argparse
import os
import sys

from .server import main


def cli() -> None:
    """Parse CLI arguments and launch the engram MCP server."""
    parser = argparse.ArgumentParser(
        description="Engram MCP memory server -- persistent three-layer memory for AI agents.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (local) or sse (network). Default: stdio.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address for SSE transport. Default: 0.0.0.0 (all interfaces).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8788,
        help="Port for SSE transport. Default: 8788.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ENGRAM_API_KEY"),
        help="Optional API key for SSE auth. Can also set ENGRAM_API_KEY env var.",
    )

    args = parser.parse_args()

    if args.transport == "sse":
        print(f"Starting engram SSE server on {args.host}:{args.port}", file=sys.stderr)
        if args.api_key:
            print("API key authentication enabled.", file=sys.stderr)
        elif args.host in ("0.0.0.0", "::"):
            print(
                "WARNING: Binding to all interfaces WITHOUT API key authentication.\n"
                "Anyone on your network can read and write memories.\n"
                "Set --api-key or ENGRAM_API_KEY to secure this endpoint.\n"
                "To bind to localhost only, use --host 127.0.0.1\n"
                "\n"
                "If exposing beyond a trusted mesh VPN (e.g. Tailscale), deploy\n"
                "behind a reverse proxy with TLS (Caddy, Nginx) to prevent\n"
                "plaintext credential sniffing.",
                file=sys.stderr,
            )
        else:
            print("No API key set.", file=sys.stderr)

    main(
        transport=args.transport,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    cli()
