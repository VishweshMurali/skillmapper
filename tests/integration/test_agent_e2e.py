"""Full agent run against real Ollama + Chroma. Slow (~20-45s per call).

Run with: uv run pytest -m ollama tests/integration
"""

import pytest

from skillmapper.agent.runtime import AgentRuntime

pytestmark = [pytest.mark.ollama, pytest.mark.slow]


@pytest.fixture(scope="module")
def runtime() -> AgentRuntime:
    return AgentRuntime()


def test_strong_match_pair_completes_without_escalation(runtime):
    result = runtime.compare_resume_to_job(
        "resume-01-senior-swe-backend", "job-01-swe-backend", force_rerun=True
    )
    assert result.status == "completed"
    assert result.verdict in ("strong_match", "partial_match", "weak_match")
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.evidence) > 0


def test_repeat_call_short_circuits_to_cached_result(runtime):
    first = runtime.compare_resume_to_job(
        "resume-01-senior-swe-backend", "job-01-swe-backend", force_rerun=True
    )
    second = runtime.compare_resume_to_job(
        "resume-01-senior-swe-backend", "job-01-swe-backend", force_rerun=False
    )
    assert second.id == first.id
