from __future__ import annotations

from pydantic import BaseModel

from skillmapper.models import EvaluationResult


class SearchResultItem(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: str
    section: str
    text: str
    score: float


class SearchDocumentsResult(BaseModel):
    results: list[SearchResultItem]


class RetrieveEvidenceResult(BaseModel):
    requirement: str
    evidence: list[SearchResultItem]


class CaptureFeedbackResult(BaseModel):
    feedback_id: str
    status: str


class LookupEvaluationsResult(BaseModel):
    results: list[EvaluationResult]
