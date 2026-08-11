from __future__ import annotations

from skillmapper.mcp_server.schemas import RetrieveEvidenceResult, SearchResultItem
from skillmapper.retrieval.hybrid import HybridRetriever


def retrieve_evidence(
    retriever: HybridRetriever,
    resume_id: str,
    job_id: str,  # kept for API symmetry with compare_resume_to_job; not used to filter
    requirement: str,
    top_k: int = 5,
) -> RetrieveEvidenceResult:
    hits = retriever.search(requirement, k=top_k, doc_id=resume_id, doc_type="resume")
    return RetrieveEvidenceResult(
        requirement=requirement,
        evidence=[
            SearchResultItem(
                chunk_id=hit["id"],
                doc_id=hit["metadata"]["doc_id"],
                doc_type=hit["metadata"]["doc_type"],
                section=hit["metadata"]["section"],
                text=hit["text"],
                score=hit["score"],
            )
            for hit in hits
        ],
    )
