from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


class ResumeSection(StrEnum):
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    SKILLS = "skills"
    EDUCATION = "education"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    OTHER = "other"


class JobSection(StrEnum):
    OVERVIEW = "overview"
    RESPONSIBILITIES = "responsibilities"
    REQUIREMENTS = "requirements"
    QUALIFICATIONS = "qualifications"
    OTHER = "other"


DocType = Literal["resume", "job_description"]


class Resume(BaseModel):
    id: str
    candidate_name: str
    source_path: str
    raw_text: str


class JobDescription(BaseModel):
    id: str
    title: str
    company: str
    source_path: str
    raw_text: str


class Chunk(BaseModel):
    """A single chunk produced by ingestion. `id` is deterministic
    (`{doc_id}::{section}::{chunk_index}`) so re-ingesting the same document
    is idempotent (upsert, not append).
    """

    id: str
    doc_id: str
    doc_type: DocType
    section: str
    chunk_index: int
    text: str

    @staticmethod
    def make_id(doc_id: str, section: str, chunk_index: int) -> str:
        return f"{doc_id}::{section}::{chunk_index}"


class EvidenceItem(BaseModel):
    chunk_id: str
    doc_id: str
    section: str
    text: str
    score: float
    supports_requirement: str


Verdict = Literal["strong_match", "partial_match", "weak_match"]
EvaluationStatus = Literal["completed", "escalated", "completed_reviewed"]


class EvaluationResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resume_id: str
    job_id: str
    verdict: Verdict
    overall_score: float
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    reasoning: str
    confidence: float
    status: EvaluationStatus
    escalation_reason: str | None = None
    retries_used: int = 0
    created_at: datetime = Field(default_factory=_now)


ReviewerDecision = Literal["approve", "override", "reject"]


class ReviewerFeedback(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evaluation_id: str
    reviewer_name: str
    decision: ReviewerDecision
    corrected_verdict: Verdict | None = None
    comments: str | None = None
    created_at: datetime = Field(default_factory=_now)
