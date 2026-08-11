from __future__ import annotations

# Same k-widening constant the LangGraph agent's `retrieve` node uses --
# single source of truth for "what k means on a retry" across both
# orchestration paths.
from skillmapper.agent.nodes import BASE_K
from skillmapper.models import EvidenceItem
from skillmapper.orchestration.schemas import DecideRetrievalResult
from skillmapper.retrieval.hybrid import HybridRetriever


def decide_retrieval(
    retriever: HybridRetriever, requirement: str, resume_id: str, retry_count: int = 0
) -> DecideRetrievalResult:
    k = BASE_K + retry_count * 3
    hits = retriever.search(requirement, k=k, doc_id=resume_id, doc_type="resume")
    evidence = [
        EvidenceItem(
            chunk_id=hit["id"],
            doc_id=hit["metadata"]["doc_id"],
            section=hit["metadata"]["section"],
            text=hit["text"],
            score=hit["score"],
            supports_requirement=requirement,
        )
        for hit in hits
    ]
    return DecideRetrievalResult(requirement=requirement, k_used=k, evidence=evidence)
