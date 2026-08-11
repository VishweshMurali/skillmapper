from __future__ import annotations

import sys
from pathlib import Path

from mcp.client.stdio import StdioServerParameters

REPO_ROOT = Path(__file__).resolve().parents[4]
MCP_SERVERS_DIR = REPO_ROOT / "mcp_servers"

# command=sys.executable (this interpreter, i.e. the shared workspace venv)
# rather than `uv run --project <dir> server.py`: all 6 new workspace
# members share one venv (see pyproject.toml [tool.uv.workspace]), so
# there's no per-member environment to select, and direct invocation avoids
# 5x `uv run` resolution overhead per orchestrator session.
_SERVER_DIR_NAMES = {
    "jd_parser": "jd-parser",
    "rag_router": "rag-router",
    "resume_comparator": "resume-comparator",
    "scorer": "scorer",
    "ragas_eval": "ragas-eval",
}

SERVER_SPECS: dict[str, StdioServerParameters] = {
    name: StdioServerParameters(command=sys.executable, args=[str(MCP_SERVERS_DIR / dir_name / "server.py")])
    for name, dir_name in _SERVER_DIR_NAMES.items()
}

# An 8B local model plus widened-k retries can run for minutes -- these are
# per-call read timeouts, not a total-session budget.
TOOL_TIMEOUTS = {
    "parse_jd": 30,
    "decide_retrieval": 60,
    "compare_resume_jd": 600,
    "score_comparison": 10,
    "run_ragas_eval": 1800,
}
