from skillmapper.agent import graph as graph_module
from skillmapper.agent import nodes
from skillmapper.agent.llm import ComparisonDraft
from skillmapper.agent.state import make_initial_state
from skillmapper.models import JobDescription, Resume

RESUME = Resume(id="r1", candidate_name="Test Candidate", source_path="x", raw_text="resume text")
JOB = JobDescription(id="j1", title="Engineer", company="Acme", source_path="y", raw_text="job text")


class FakeCollection:
    def get(self, where, include):
        return {"documents": ["Must know Python."], "metadatas": [{"section": "requirements"}]}


class ScoreByWidthRetriever:
    """Returns a below-threshold evidence score at the base k, and an
    above-threshold score once k has been widened by a retry -- models a
    retry that legitimately finds better evidence, not a repeated no-op.
    """

    def __init__(self, base_k: int, always_low: bool = False):
        self.collection = FakeCollection()
        self.base_k = base_k
        self.always_low = always_low
        self.k_calls: list[int] = []

    def search(self, query, k, doc_id, doc_type):
        self.k_calls.append(k)
        score = 0.0 if (self.always_low or k <= self.base_k) else 0.03
        return [
            {
                "id": "r1::skills::0",
                "text": "Python expert",
                "metadata": {"doc_id": doc_id, "section": "skills"},
                "score": score,
            }
        ]


def _stub_generate_comparison(monkeypatch):
    def fake(llm, prompt, max_parse_retries=2):
        return ComparisonDraft(
            verdict="partial_match",
            overall_score=0.5,
            matched_skills=["Python"],
            missing_skills=[],
            reasoning="stub reasoning",
            self_reported_confidence=0.5,
        )

    monkeypatch.setattr(nodes, "generate_comparison", fake)


def _stub_repository(monkeypatch):
    saved = []
    monkeypatch.setattr(nodes.repository, "save_evaluation", lambda result: saved.append(result))
    return saved


def test_retry_widens_k_and_then_finalizes(monkeypatch):
    _stub_generate_comparison(monkeypatch)
    saved = _stub_repository(monkeypatch)

    retriever = ScoreByWidthRetriever(base_k=5)
    compiled = graph_module.build_graph(
        retriever, llm=None, resumes_by_id={"r1": RESUME}, jobs_by_id={"j1": JOB},
        confidence_threshold=0.65,
    )
    initial = make_initial_state("r1", "j1", max_retries=2)
    final_state = graph_module.invoke_graph(compiled, initial)

    assert final_state["status"] == "completed"
    assert final_state["retry_count"] == 1
    # k widened on the retry rather than repeating the same call
    assert retriever.k_calls == [5, 8]
    assert len(saved) == 1
    assert saved[0].status == "completed"
    assert saved[0].retries_used == 1


def test_escalates_after_max_retries_when_confidence_never_recovers(monkeypatch):
    _stub_generate_comparison(monkeypatch)
    saved = _stub_repository(monkeypatch)

    retriever = ScoreByWidthRetriever(base_k=5, always_low=True)
    compiled = graph_module.build_graph(
        retriever, llm=None, resumes_by_id={"r1": RESUME}, jobs_by_id={"j1": JOB},
        confidence_threshold=0.65,
    )
    initial = make_initial_state("r1", "j1", max_retries=2)
    final_state = graph_module.invoke_graph(compiled, initial)

    assert final_state["status"] == "escalated"
    assert final_state["retry_count"] == 2
    assert final_state["escalation_reason"] is not None
    # base k=5, then widened twice: 5, 8, 11 -- three genuinely distinct attempts
    assert retriever.k_calls == [5, 8, 11]
    assert len(saved) == 1
    assert saved[0].status == "escalated"
    assert saved[0].retries_used == 2
