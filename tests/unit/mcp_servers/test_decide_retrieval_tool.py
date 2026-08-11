from decide_retrieval_tool import decide_retrieval


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, k, doc_type=None, doc_id=None, section=None):
        self.calls.append(dict(query=query, k=k, doc_type=doc_type, doc_id=doc_id, section=section))
        return self.hits


def _hit(id_, score=0.03, doc_id="resume-01", section="skills", text="Python and Kubernetes"):
    return {"id": id_, "text": text, "score": score, "metadata": {"doc_id": doc_id, "doc_type": "resume", "section": section}}


def test_decide_retrieval_uses_base_k_on_first_attempt():
    retriever = FakeRetriever([_hit("c1")])

    result = decide_retrieval(retriever, "Must know Python", "resume-01", retry_count=0)

    assert retriever.calls[0]["k"] == 5
    assert retriever.calls[0]["doc_id"] == "resume-01"
    assert retriever.calls[0]["doc_type"] == "resume"
    assert result.k_used == 5
    assert result.evidence[0].chunk_id == "c1"
    assert result.evidence[0].supports_requirement == "Must know Python"


def test_decide_retrieval_widens_k_on_retry():
    retriever = FakeRetriever([])

    result = decide_retrieval(retriever, "Must know Python", "resume-01", retry_count=2)

    assert retriever.calls[0]["k"] == 5 + 2 * 3
    assert result.k_used == 11


def test_decide_retrieval_maps_hit_metadata_into_evidence_items():
    retriever = FakeRetriever([_hit("c1", score=0.02, doc_id="resume-02", section="experience")])

    result = decide_retrieval(retriever, "5 years experience", "resume-02")

    item = result.evidence[0]
    assert item.doc_id == "resume-02"
    assert item.section == "experience"
    assert item.score == 0.02
