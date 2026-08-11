from __future__ import annotations

from skillmapper.mcp_server.schemas import LookupEvaluationsResult
from skillmapper.storage import repository


def lookup_evaluation_result(
    evaluation_id: str | None = None,
    resume_id: str | None = None,
    job_id: str | None = None,
    status: str | None = None,
) -> LookupEvaluationsResult:
    if evaluation_id is not None:
        result = repository.get_evaluation(evaluation_id)
        return LookupEvaluationsResult(results=[result] if result else [])

    results = repository.query_evaluations(resume_id=resume_id, job_id=job_id, status=status)
    return LookupEvaluationsResult(results=results)
