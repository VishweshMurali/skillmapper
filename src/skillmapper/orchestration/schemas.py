from __future__ import annotations

from pydantic import BaseModel

from skillmapper.models import EvidenceItem, Verdict


class RequirementBlock(BaseModel):
    section: str
    heading: str
    text: str


class ParseJdResult(BaseModel):
    job_id: str
    title: str
    company: str
    sections: list[RequirementBlock]
    requirement_blocks: list[RequirementBlock]


class DecideRetrievalResult(BaseModel):
    requirement: str
    k_used: int
    evidence: list[EvidenceItem]


class CompareResult(BaseModel):
    resume_id: str
    job_id: str
    verdict: Verdict
    overall_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    reasoning: str
    self_reported_confidence: float


class ScoreResult(BaseModel):
    confidence: float
    evidence_coverage: float
    self_reported_confidence: float


class RagasEvalResult(BaseModel):
    session_id: str
    sample_count: int
    scores: dict[str, float]
    metric_names: list[str]


class EvalLogsResource(BaseModel):
    session_id: str
    event_count: int
    events: list[dict]
