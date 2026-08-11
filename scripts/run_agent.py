"""Run the LangGraph agent end to end against real Ollama + Chroma.

Usage:
    uv run scripts/run_agent.py <resume_id> <job_id> [--force-rerun]

Example:
    uv run scripts/run_agent.py resume-01-senior-swe-backend job-01-swe-backend
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmapper.agent.runtime import AgentRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("resume_id")
    parser.add_argument("job_id")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    runtime = AgentRuntime()
    start = time.perf_counter()
    result = runtime.compare_resume_to_job(args.resume_id, args.job_id, force_rerun=args.force_rerun)
    elapsed = time.perf_counter() - start

    print(f"\n=== EvaluationResult ({elapsed:.1f}s) ===")
    print(f"status:        {result.status}")
    print(f"verdict:       {result.verdict}")
    print(f"overall_score: {result.overall_score}")
    print(f"confidence:    {result.confidence}")
    print(f"retries_used:  {result.retries_used}")
    print(f"matched:       {result.matched_skills}")
    print(f"missing:       {result.missing_skills}")
    print(f"reasoning:     {result.reasoning}")
    if result.escalation_reason:
        print(f"escalation:    {result.escalation_reason}")
    print(f"evidence_items: {len(result.evidence)}")


if __name__ == "__main__":
    main()
