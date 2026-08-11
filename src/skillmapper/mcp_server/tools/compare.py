from __future__ import annotations

from skillmapper.agent.runtime import AgentRuntime
from skillmapper.models import EvaluationResult


def compare_resume_to_job(
    runtime: AgentRuntime, resume_id: str, job_id: str, force_rerun: bool = False
) -> EvaluationResult:
    return runtime.compare_resume_to_job(resume_id, job_id, force_rerun=force_rerun)
