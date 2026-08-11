"""Run the 5-server MCP orchestrator end to end (spawns rag-router,
jd-parser, resume-comparator, scorer as stdio subprocesses and runs the
same retry/escalation loop the LangGraph agent implements in-process).

Usage:
    uv run scripts/run_orchestrator.py <resume_id> <job_id> [--force-rerun]

Example:
    uv run scripts/run_orchestrator.py resume-01-senior-swe-backend job-01-swe-backend
"""

from __future__ import annotations

from skillmapper_orchestrator.cli import main

if __name__ == "__main__":
    main()
