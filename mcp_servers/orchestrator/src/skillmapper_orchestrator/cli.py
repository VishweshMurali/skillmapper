from __future__ import annotations

import argparse
import asyncio
import time

from skillmapper.models import EvaluationResult
from skillmapper.storage import repository

from skillmapper_orchestrator.client import connect_all
from skillmapper_orchestrator.retry_loop import run_session
from skillmapper_orchestrator.server_specs import TOOL_TIMEOUTS


async def run(
    resume_id: str, job_id: str, force_rerun: bool = False, eval_after: bool = False
) -> tuple[str | None, EvaluationResult, dict | None]:
    """Behavioral parity with AgentRuntime.compare_resume_to_job: a cached
    completed evaluation short-circuits the whole 5-server round trip unless
    force_rerun is set. session_id is None when served from cache -- no new
    orchestrator session was run, so eval_after is skipped in that case
    (nothing new was logged to score).

    `eval_after` calls ragas-eval on the just-finished session through the
    same already-open connection connect_all() paid for -- otherwise the
    orchestrator spawns that 5th server on every run without ever using it.
    """
    if not force_rerun:
        existing = repository.find_completed_evaluation(resume_id, job_id)
        if existing is not None:
            return None, existing, None

    async with connect_all() as clients:
        session_id, result = await run_session(clients, resume_id, job_id)
        ragas_scores = None
        if eval_after:
            ragas_scores = await clients.call_tool(
                "ragas_eval",
                "run_ragas_eval",
                {"session_id": session_id},
                timeout=TOOL_TIMEOUTS["run_ragas_eval"],
            )
        return session_id, result, ragas_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("resume_id")
    parser.add_argument("job_id")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--eval", action="store_true",
        help="Also RAGAS-score the resulting session (faithfulness + answer_relevancy).",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    session_id, result, ragas_scores = asyncio.run(
        run(args.resume_id, args.job_id, force_rerun=args.force_rerun, eval_after=args.eval)
    )
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
    if session_id:
        print(f"session_id:    {session_id}")
    else:
        print("session_id:    (none -- served from cache)")
        if args.eval:
            print("ragas_eval:    skipped (no fresh session to score; pass --force-rerun too)")
    if ragas_scores is not None:
        print(f"ragas_scores:  {ragas_scores['scores']}")


if __name__ == "__main__":
    main()
