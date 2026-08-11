from __future__ import annotations

from typing import Literal, TypedDict

from skillmapper.models import EvaluationResult, EvidenceItem


class AgentState(TypedDict):
    resume_id: str
    job_id: str
    retrieved_evidence: list[EvidenceItem]
    comparison_draft: dict | None
    confidence: float
    retry_count: int
    max_retries: int
    status: Literal["pending", "completed", "escalated"]
    escalation_reason: str | None
    final_result: EvaluationResult | None


def make_initial_state(resume_id: str, job_id: str, max_retries: int) -> AgentState:
    return AgentState(
        resume_id=resume_id,
        job_id=job_id,
        retrieved_evidence=[],
        comparison_draft=None,
        confidence=0.0,
        retry_count=0,
        max_retries=max_retries,
        status="pending",
        escalation_reason=None,
        final_result=None,
    )
