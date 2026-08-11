"""Entry point for running the SkillMapper MCP server over stdio.

Usage:
    uv run scripts/run_mcp_server.py

Or point an MCP client (e.g. MCP Inspector, Claude Desktop) at:
    uv run --project E:\\Skillmapper2.0 python scripts/run_mcp_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmapper.mcp_server.server import main

if __name__ == "__main__":
    main()
