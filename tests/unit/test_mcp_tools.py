from skillmapper.mcp_server.tools import compare as compare_tool
from skillmapper.mcp_server.tools import feedback as feedback_tool
from skillmapper.mcp_server.tools import lookup as lookup_tool
from skillmapper.mcp_server.tools import retrieve_evidence as retrieve_evidence_tool
from skillmapper.mcp_server.tools import search_documents as search_tool
from skillmapper.models import EvaluationResult, ReviewerFeedback


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, k, doc_type=None, doc_id=None, section=None):
        self.calls.append(dict(query=query, k=k, doc_type=doc_type, doc_id=doc_id, section=section))
        return self.hits


def _hit(id_, score=0.5, doc_id="resume-01", doc_type="resume", section="skills", text="Python"):
    return {"id": id_, "text": text, "score": score, "metadata": {"doc_id": doc_id, "doc_type": doc_type, "section": section}}


def test_search_documents_maps_hits_to_schema():
    retriever = FakeRetriever([_hit("c1")])
    result = search_tool.search_documents(retriever, "python", doc_type="resume", top_k=3)
    assert result.results[0].chunk_id == "c1"
    assert result.results[0].section == "skills"
    assert retriever.calls[0]["doc_type"] == "resume"


def test_search_documents_any_doc_type_passes_no_filter():
    retriever = FakeRetriever([])
    search_tool.search_documents(retriever, "python", doc_type="any")
    assert retriever.calls[0]["doc_type"] is None


def test_retrieve_evidence_filters_to_given_resume():
    retriever = FakeRetriever([_hit("c1", doc_id="resume-02")])
    result = retrieve_evidence_tool.retrieve_evidence(retriever, "resume-02", "job-01", "Must know Python")
    assert retriever.calls[0]["doc_id"] == "resume-02"
    assert retriever.calls[0]["doc_type"] == "resume"
    assert result.requirement == "Must know Python"
    assert result.evidence[0].doc_id == "resume-02"


class FakeRuntime:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def compare_resume_to_job(self, resume_id, job_id, force_rerun=False):
        self.calls.append((resume_id, job_id, force_rerun))
        return self.result


def _sample_result(**overrides):
    defaults = dict(
        resume_id="resume-01", job_id="job-01", verdict="strong_match", overall_score=0.9,
        reasoning="ok", confidence=0.9, status="completed",
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def test_compare_tool_delegates_to_runtime():
    runtime = FakeRuntime(_sample_result())
    result = compare_tool.compare_resume_to_job(runtime, "resume-01", "job-01", force_rerun=True)
    assert result.verdict == "strong_match"
    assert runtime.calls == [("resume-01", "job-01", True)]


def test_capture_feedback_rejects_unknown_evaluation(monkeypatch):
    monkeypatch.setattr(feedback_tool.repository, "get_evaluation", lambda eid: None)
    try:
        feedback_tool.capture_reviewer_feedback("missing-id", "Alex", "approve")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_capture_feedback_saves_and_returns_id(monkeypatch):
    monkeypatch.setattr(feedback_tool.repository, "get_evaluation", lambda eid: _sample_result())
    saved = []
    monkeypatch.setattr(feedback_tool.repository, "save_feedback", lambda fb: saved.append(fb))

    result = feedback_tool.capture_reviewer_feedback("eval-1", "Alex", "approve", comments="looks right")
    assert result.status == "saved"
    assert len(saved) == 1
    assert isinstance(saved[0], ReviewerFeedback)
    assert saved[0].reviewer_name == "Alex"


def test_lookup_by_id_returns_single_result(monkeypatch):
    monkeypatch.setattr(lookup_tool.repository, "get_evaluation", lambda eid: _sample_result())
    result = lookup_tool.lookup_evaluation_result(evaluation_id="eval-1")
    assert len(result.results) == 1


def test_lookup_by_id_missing_returns_empty(monkeypatch):
    monkeypatch.setattr(lookup_tool.repository, "get_evaluation", lambda eid: None)
    result = lookup_tool.lookup_evaluation_result(evaluation_id="missing")
    assert result.results == []


def test_lookup_by_filters_delegates_to_query(monkeypatch):
    captured = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        return [_sample_result()]

    monkeypatch.setattr(lookup_tool.repository, "query_evaluations", fake_query)
    result = lookup_tool.lookup_evaluation_result(status="escalated")
    assert captured["status"] == "escalated"
    assert len(result.results) == 1
