from __future__ import annotations

import logging
from typing import Any

from langchain_ollama import ChatOllama

from skillmapper.agent.llm import generate_comparison
from skillmapper.agent.state import AgentState
from skillmapper.models import EvaluationResult, EvidenceItem, JobDescription, Resume
from skillmapper.retrieval.hybrid import HybridRetriever
from skillmapper.storage import repository

logger = logging.getLogger(__name__)

REQUIREMENT_SECTIONS = {"requirements", "qualifications"}
BASE_K = 5
MIN_EVIDENCE_SCORE = 0.02  # roughly a rank-1-in-at-least-one-list RRF score


def _requirement_texts(retriever: HybridRetriever, job_id: str) -> list[str]:
    raw = retriever.collection.get(
        where={"$and": [{"doc_id": job_id}, {"doc_type": "job_description"}]},
        include=["documents", "metadatas"],
    )
    texts = [
        doc
        for doc, meta in zip(raw["documents"], raw["metadatas"])
        if meta.get("section") in REQUIREMENT_SECTIONS
    ]
    return texts or raw["documents"]


def make_retrieve_node(retriever: HybridRetriever):
    def retrieve(state: AgentState) -> dict[str, Any]:
        requirements = _requirement_texts(retriever, state["job_id"])
        # widen k on each retry -- a genuinely different attempt, not a repeat
        k = BASE_K + state["retry_count"] * 3

        evidence: list[EvidenceItem] = []
        for requirement in requirements:
            hits = retriever.search(
                requirement, k=k, doc_id=state["resume_id"], doc_type="resume"
            )
            for hit in hits:
                evidence.append(
                    EvidenceItem(
                        chunk_id=hit["id"],
                        doc_id=hit["metadata"]["doc_id"],
                        section=hit["metadata"]["section"],
                        text=hit["text"],
                        score=hit["score"],
                        supports_requirement=requirement,
                    )
                )
        logger.info(
            "retrieve: resume=%s job=%s retry=%d k=%d evidence_items=%d",
            state["resume_id"], state["job_id"], state["retry_count"], k, len(evidence),
        )
        return {"retrieved_evidence": evidence}

    return retrieve


def make_compare_node(
    llm: ChatOllama, resumes_by_id: dict[str, Resume], jobs_by_id: dict[str, JobDescription]
):
    def compare(state: AgentState) -> dict[str, Any]:
        job = jobs_by_id[state["job_id"]]
        resume = resumes_by_id[state["resume_id"]]

        evidence_text = "\n".join(
            f"- [{e.section}] {e.text}" for e in state["retrieved_evidence"]
        ) or "(no evidence retrieved)"

        prompt = f"""You are evaluating whether a candidate's resume matches a job description.

Job Title: {job.title} at {job.company}
Job Description:
{job.raw_text}

Candidate: {resume.candidate_name}
Retrieved resume evidence (via hybrid search against the job's requirements):
{evidence_text}

Based ONLY on the evidence above, evaluate the match and respond with a JSON object with fields:
- verdict: one of "strong_match", "partial_match", "weak_match"
- overall_score: float between 0 and 1
- matched_skills: list of requirements/skills clearly supported by the evidence
- missing_skills: list of requirements not supported by the evidence
- reasoning: 2-4 sentence explanation citing the evidence
- self_reported_confidence: float between 0 and 1, your own confidence in this assessment
"""
        draft = generate_comparison(llm, prompt)
        logger.info(
            "compare: resume=%s job=%s verdict=%s self_reported_confidence=%.2f",
            state["resume_id"], state["job_id"], draft.verdict, draft.self_reported_confidence,
        )
        return {"comparison_draft": draft.model_dump()}

    return compare


def confidence_check(state: AgentState) -> dict[str, Any]:
    draft = state["comparison_draft"] or {}
    evidence = state["retrieved_evidence"]

    by_requirement: dict[str, list[EvidenceItem]] = {}
    for item in evidence:
        by_requirement.setdefault(item.supports_requirement, []).append(item)

    if by_requirement:
        covered = sum(
            1
            for items in by_requirement.values()
            if any(i.score >= MIN_EVIDENCE_SCORE for i in items)
        )
        evidence_coverage = covered / len(by_requirement)
    else:
        evidence_coverage = 0.0

    # Evidence-coverage-dominant: an 8B model's self-reported confidence is
    # close to uncalibrated noise, so it's kept but weighted lightly.
    self_reported = draft.get("self_reported_confidence", 0.0)
    confidence = 0.3 * self_reported + 0.7 * evidence_coverage

    logger.info(
        "confidence_check: resume=%s job=%s coverage=%.2f self_reported=%.2f confidence=%.2f",
        state["resume_id"], state["job_id"], evidence_coverage, self_reported, confidence,
    )
    return {"confidence": confidence}


def increment_retry(state: AgentState) -> dict[str, Any]:
    return {"retry_count": state["retry_count"] + 1}


def make_finalize_node():
    def finalize(state: AgentState) -> dict[str, Any]:
        draft = state["comparison_draft"] or {}
        result = EvaluationResult(
            resume_id=state["resume_id"],
            job_id=state["job_id"],
            verdict=draft.get("verdict", "weak_match"),
            overall_score=draft.get("overall_score", 0.0),
            matched_skills=draft.get("matched_skills", []),
            missing_skills=draft.get("missing_skills", []),
            evidence=state["retrieved_evidence"],
            reasoning=draft.get("reasoning", ""),
            confidence=state["confidence"],
            status="completed",
            retries_used=state["retry_count"],
        )
        repository.save_evaluation(result)
        return {"final_result": result, "status": "completed"}

    return finalize


def make_escalate_node():
    def escalate(state: AgentState) -> dict[str, Any]:
        draft = state["comparison_draft"] or {}
        reason = (
            f"confidence {state['confidence']:.2f} remained below threshold "
            f"after {state['retry_count']} retries"
        )
        result = EvaluationResult(
            resume_id=state["resume_id"],
            job_id=state["job_id"],
            verdict=draft.get("verdict", "weak_match"),
            overall_score=draft.get("overall_score", 0.0),
            matched_skills=draft.get("matched_skills", []),
            missing_skills=draft.get("missing_skills", []),
            evidence=state["retrieved_evidence"],
            reasoning=draft.get("reasoning", "") or "Escalated for human review.",
            confidence=state["confidence"],
            status="escalated",
            escalation_reason=reason,
            retries_used=state["retry_count"],
        )
        repository.save_evaluation(result)
        return {"final_result": result, "status": "escalated", "escalation_reason": reason}

    return escalate
