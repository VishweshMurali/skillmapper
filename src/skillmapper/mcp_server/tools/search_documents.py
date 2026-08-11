from __future__ import annotations

from skillmapper.mcp_server.schemas import SearchDocumentsResult, SearchResultItem
from skillmapper.retrieval.hybrid import HybridRetriever


def search_documents(
    retriever: HybridRetriever,
    query: str,
    doc_type: str = "any",
    section: str | None = None,
    top_k: int = 5,
) -> SearchDocumentsResult:
    filter_doc_type = None if doc_type == "any" else doc_type
    hits = retriever.search(query, k=top_k, doc_type=filter_doc_type, section=section)
    return SearchDocumentsResult(
        results=[
            SearchResultItem(
                chunk_id=hit["id"],
                doc_id=hit["metadata"]["doc_id"],
                doc_type=hit["metadata"]["doc_type"],
                section=hit["metadata"]["section"],
                text=hit["text"],
                score=hit["score"],
            )
            for hit in hits
        ]
    )
