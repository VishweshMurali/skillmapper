from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

from mcp.server.mcpserver import MCPServer
from score_comparison_tool import score_comparison as _score_comparison

from skillmapper.models import EvidenceItem
from skillmapper.orchestration import session_log
from skillmapper.orchestration.schemas import ScoreResult

mcp = MCPServer("scorer")


@mcp.tool()
def score_comparison(
    evidence: list[EvidenceItem],
    self_reported_confidence: float,
    session_id: str,
    turn_index: int,
) -> ScoreResult:
    """Evidence-coverage-weighted confidence score for one comparison attempt."""
    result = _score_comparison(evidence, self_reported_confidence)
    session_log.append_event(
        session_log.SessionEvent(
            session_id=session_id,
            turn_index=turn_index,
            event="score_comparison",
            payload=result.model_dump(mode="json"),
        )
    )
    return result


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
