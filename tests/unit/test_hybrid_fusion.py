from skillmapper.retrieval.hybrid import (
    HybridRetriever,
    _matches_filter,
    _reciprocal_rank_fusion,
)


def test_rrf_rewards_items_ranked_highly_in_both_lists():
    dense = ["a", "b", "c"]
    bm25 = ["b", "a", "d"]
    scores = _reciprocal_rank_fusion([dense, bm25])
    # "a" is rank0 dense + rank1 bm25; "b" is rank1 dense + rank0 bm25 -> tied for top
    assert scores["a"] == scores["b"]
    assert scores["a"] > scores["c"]
    assert scores["a"] > scores["d"]


def test_rrf_item_only_in_one_list_still_scores() :
    scores = _reciprocal_rank_fusion([["a", "b"], []])
    assert scores["a"] > scores["b"] > 0


def test_matches_filter_all_none_matches_everything():
    assert _matches_filter({"doc_type": "resume", "doc_id": "x", "section": "skills"}, None, None, None)


def test_matches_filter_rejects_wrong_doc_id():
    meta = {"doc_type": "resume", "doc_id": "resume-01", "section": "skills"}
    assert not _matches_filter(meta, None, "resume-02", None)
    assert _matches_filter(meta, None, "resume-01", None)


class FakeCollection:
    """Mimics the subset of chromadb.Collection used by vector_store.query."""

    def __init__(self, hits: list[dict]):
        self._hits = hits

    def query(self, query_embeddings, n_results, where=None, include=None):
        return {
            "ids": [[h["id"] for h in self._hits]],
            "documents": [[h["text"] for h in self._hits]],
            "metadatas": [[h["metadata"] for h in self._hits]],
            "distances": [[h["distance"] for h in self._hits]],
        }


class FakeBM25Index:
    def __init__(self, scored: list[tuple[dict, float]]):
        self._scored = scored

    def search(self, query):
        return self._scored


class FakeEmbedder:
    def embed(self, text, kind):
        return [0.0]


def test_hybrid_search_excludes_unfiltered_bm25_hits_from_a_different_doc():
    dense_hits = [
        {"id": "resume-01::skills::0", "text": "Python", "metadata": {"doc_type": "resume", "doc_id": "resume-01", "section": "skills"}, "distance": 0.1},
    ]
    bm25_scored = [
        ({"id": "resume-01::skills::0", "text": "Python", "metadata": {"doc_type": "resume", "doc_id": "resume-01", "section": "skills"}}, 5.0),
        # different doc_id -- must be excluded once we filter to doc_id="resume-01"
        ({"id": "resume-02::skills::0", "text": "Python expert", "metadata": {"doc_type": "resume", "doc_id": "resume-02", "section": "skills"}}, 9.0),
    ]

    retriever = HybridRetriever(
        collection=FakeCollection(dense_hits),
        bm25_index=FakeBM25Index(bm25_scored),
        embedder=FakeEmbedder(),
    )

    results = retriever.search("python", k=5, doc_id="resume-01")
    ids = [r["id"] for r in results]
    assert "resume-02::skills::0" not in ids
    assert "resume-01::skills::0" in ids
