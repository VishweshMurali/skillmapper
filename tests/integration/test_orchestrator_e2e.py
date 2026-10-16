"""Full 5-server MCP orchestrator run: spawns rag-router, jd-parser,
resume-comparator, scorer, ragas-eval as real stdio subprocesses against
real Ollama + Chroma. Slow -- mirrors tests/integration/test_agent_e2e.py.

Run with: uv run pytest -m "ollama and slow" tests/integration
"""

from __future__ import annotations

import pytest
from skillmapper_orchestrator.cli import run

from skillmapper.orchestration.session_log import read_session, session_log_path

pytestmark = [pytest.mark.ollama, pytest.mark.slow]


async def test_orchestrator_run_completes_and_logs_a_ragas_ready_session():
    session_id, result, ragas_scores = await run(
        "resume-01-senior-swe-backend", "job-01-swe-backend", force_rerun=True, eval_after=True
    )

    assert session_id is not None
    assert result.status in ("completed", "escalated")
    assert result.verdict in ("strong_match", "partial_match", "weak_match")
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.evidence) > 0

    events = read_session(session_id)
    assert any(e.event == "parse_jd" for e in events)
    assert any(e.event == "decide_retrieval" for e in events)
    compare_events = [e for e in events if e.event == "compare_resume_jd"]
    assert len(compare_events) == result.retries_used + 1
    assert compare_events[-1].response == result.reasoning

    # eval_after=True calls ragas-eval on this exact session over the same
    # already-open connection -- the orchestrator's 5th server actually used.
    assert ragas_scores is not None
    assert ragas_scores["session_id"] == session_id
    assert set(ragas_scores["scores"]) == {"faithfulness", "answer_relevancy"}

    session_log_path(session_id).unlink(missing_ok=True)


async def test_orchestrator_repeat_call_short_circuits_to_cached_result():
    first_session_id, first, _ = await run(
        "resume-01-senior-swe-backend", "job-01-swe-backend", force_rerun=True
    )
    second_session_id, second, ragas_scores = await run(
        "resume-01-senior-swe-backend", "job-01-swe-backend", force_rerun=False
    )

    assert second.id == first.id
    assert second_session_id is None
    assert ragas_scores is None

    session_log_path(first_session_id).unlink(missing_ok=True)
