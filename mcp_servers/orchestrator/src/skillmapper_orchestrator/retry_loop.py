from __future__ import annotations

import uuid

from skillmapper.config import settings
from skillmapper.models import EvaluationResult
from skillmapper.storage import repository
from skillmapper_orchestrator.client import OrchestratorClients
from skillmapper_orchestrator.server_specs import TOOL_TIMEOUTS


async def run_session(
    clients: OrchestratorClients,
    resume_id: str,
    job_id: str,
    threshold: float | None = None,
    max_retries: int | None = None,
) -> tuple[str, EvaluationResult]:
    """Direct semantic port of agent/graph.py:route_after_confidence_check +
    agent/nodes.py's retrieve -> compare -> confidence_check cycle, now
    driven by MCP tool calls instead of LangGraph node returns. Evidence is
    fully recomputed each retry (not accumulated across retries), matching
    make_retrieve_node's behavior.
    """
    session_id = str(uuid.uuid4())
    threshold = threshold if threshold is not None else settings.agent_confidence_threshold
    max_retries = max_retries if max_retries is not None else settings.agent_max_retries

    turn_index = 0
    jd = await clients.call_tool(
        "jd_parser",
        "parse_jd",
        {"job_id": job_id, "session_id": session_id, "turn_index": turn_index},
        timeout=TOOL_TIMEOUTS["parse_jd"],
    )
    turn_index += 1
    requirement_blocks = jd["requirement_blocks"]

    retry_count = 0
    cmp: dict | None = None
    score: dict | None = None
    evidence: list[dict] = []
    status: str
    escalation_reason: str | None

    while True:
        evidence = []
        for block in requirement_blocks:
            dr = await clients.call_tool(
                "rag_router",
                "decide_retrieval",
                {
                    "requirement": block["text"],
                    "resume_id": resume_id,
                    "retry_count": retry_count,
                    "session_id": session_id,
                    "turn_index": turn_index,
                },
                timeout=TOOL_TIMEOUTS["decide_retrieval"],
            )
            turn_index += 1
            evidence.extend(dr["evidence"])

        cmp = await clients.call_tool(
            "resume_comparator",
            "compare_resume_jd",
            {
                "resume_id": resume_id,
                "job_id": job_id,
                "evidence": evidence,
                "session_id": session_id,
                "turn_index": turn_index,
            },
            timeout=TOOL_TIMEOUTS["compare_resume_jd"],
        )
        turn_index += 1

        score = await clients.call_tool(
            "scorer",
            "score_comparison",
            {
                "evidence": evidence,
                "self_reported_confidence": cmp["self_reported_confidence"],
                "session_id": session_id,
                "turn_index": turn_index,
            },
            timeout=TOOL_TIMEOUTS["score_comparison"],
        )
        turn_index += 1

        if score["confidence"] >= threshold:
            status = "completed"
            escalation_reason = None
            break
        if retry_count >= max_retries:
            status = "escalated"
            escalation_reason = (
                f"confidence {score['confidence']:.2f} remained below threshold "
                f"after {retry_count} retries"
            )
            break
        retry_count += 1

    result = EvaluationResult(
        resume_id=resume_id,
        job_id=job_id,
        verdict=cmp["verdict"],
        overall_score=cmp["overall_score"],
        matched_skills=cmp["matched_skills"],
        missing_skills=cmp["missing_skills"],
        evidence=evidence,
        reasoning=cmp["reasoning"],
        confidence=score["confidence"],
        status=status,
        escalation_reason=escalation_reason,
        retries_used=retry_count,
    )
    repository.save_evaluation(result)
    return session_id, result
