from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

from eval_logs_resource import eval_logs as _eval_logs
from mcp.server.mcpserver import MCPServer
from run_ragas_eval_tool import run_ragas_eval as _run_ragas_eval

from skillmapper.orchestration.schemas import EvalLogsResource, RagasEvalResult

mcp = MCPServer("ragas-eval")


@mcp.tool()
def run_ragas_eval(session_id: str, metric_names: list[str] | None = None) -> RagasEvalResult:
    """RAGAS-score one orchestrator session's logged compare_resume_jd turns.

    Defaults to faithfulness + answer_relevancy (reference-free). Requesting
    context_precision/context_recall raises if the session has no `reference`
    values, since live comparisons have no ground truth.
    """
    return _run_ragas_eval(session_id, metric_names)


@mcp.resource("session://{session_id}/events")
def eval_logs(session_id: str) -> EvalLogsResource:
    """Raw logged events for one orchestrator session."""
    return _eval_logs(session_id)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
