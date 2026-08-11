from __future__ import annotations

from typing import Any

from chromadb.api.models.Collection import Collection
from langsmith import traceable

from skillmapper.retrieval import vector_store
from skillmapper.retrieval.bm25_index import BM25Index
from skillmapper.retrieval.embeddings import Embedder

RRF_K = 60
DENSE_CANDIDATES = 50
BM25_CANDIDATES = 50


def _build_where(
    doc_type: str | None, doc_id: str | None, section: str | None
) -> dict[str, Any] | None:
    clauses = []
    if doc_type is not None:
        clauses.append({"doc_type": doc_type})
    if doc_id is not None:
        clauses.append({"doc_id": doc_id})
    if section is not None:
        clauses.append({"section": section})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _matches_filter(
    metadata: dict[str, Any], doc_type: str | None, doc_id: str | None, section: str | None
) -> bool:
    if doc_type is not None and metadata.get("doc_type") != doc_type:
        return False
    if doc_id is not None and metadata.get("doc_id") != doc_id:
        return False
    if section is not None and metadata.get("section") != section:
        return False
    return True


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = RRF_K
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, item_id in enumerate(ranked_ids):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    """Fuses Chroma dense retrieval with a BM25 keyword index via
    Reciprocal Rank Fusion, so exact keyword matches (certification names,
    specific tools) surface alongside semantically-similar results.
    """

    def __init__(self, collection: Collection, bm25_index: BM25Index, embedder: Embedder):
        self.collection = collection
        self.bm25_index = bm25_index
        self.embedder = embedder

    @traceable(name="hybrid_retrieve", run_type="retriever")
    def search(
        self,
        query: str,
        k: int = 8,
        doc_type: str | None = None,
        doc_id: str | None = None,
        section: str | None = None,
    ) -> list[dict[str, Any]]:
        where = _build_where(doc_type, doc_id, section)
        query_embedding = self.embedder.embed(query, "query")

        dense_hits = vector_store.query(
            self.collection, query_embedding, n_results=DENSE_CANDIDATES, where=where
        )
        by_id: dict[str, dict[str, Any]] = {h["id"]: h for h in dense_hits}
        dense_ranked_ids = [h["id"] for h in dense_hits]

        # BM25 has no native metadata filtering: score the whole corpus,
        # then post-filter before it's allowed to influence fusion.
        bm25_scored = self.bm25_index.search(query)
        bm25_filtered = [
            (entry, score)
            for entry, score in bm25_scored
            if _matches_filter(entry["metadata"], doc_type, doc_id, section) and score > 0
        ][:BM25_CANDIDATES]
        bm25_ranked_ids = [entry["id"] for entry, _ in bm25_filtered]
        for entry, _ in bm25_filtered:
            by_id.setdefault(entry["id"], {"id": entry["id"], "text": entry["text"], "metadata": entry["metadata"]})

        fused_scores = _reciprocal_rank_fusion([dense_ranked_ids, bm25_ranked_ids])
        ranked = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)[:k]

        return [{**by_id[item_id], "score": score} for item_id, score in ranked]
