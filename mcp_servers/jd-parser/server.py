from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")

# stdio transport requires stdout stay clean for the JSON-RPC channel --
# all logging goes to stderr, never print().
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

from mcp.server.mcpserver import MCPServer
from parse_jd_tool import parse_jd as _parse_jd

from skillmapper.config import settings
from skillmapper.orchestration import session_log
from skillmapper.orchestration.schemas import ParseJdResult

mcp = MCPServer("jd-parser")


@mcp.tool()
def parse_jd(job_id: str, session_id: str, turn_index: int) -> ParseJdResult:
    """Section-split a job description into requirement blocks."""
    result = _parse_jd(settings.synthetic_jobs_dir, job_id)
    session_log.append_event(
        session_log.SessionEvent(
            session_id=session_id,
            turn_index=turn_index,
            event="parse_jd",
            job_id=job_id,
            payload=result.model_dump(mode="json"),
        )
    )
    return result


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
