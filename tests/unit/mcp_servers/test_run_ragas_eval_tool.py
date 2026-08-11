import pytest
import run_ragas_eval_tool
from run_ragas_eval_tool import run_ragas_eval

from skillmapper.orchestration.session_log import SessionEvent, append_event


class FakeRagasResult:
    def __init__(self, scores: dict[str, list[float]]):
        self._scores = scores

    def __getitem__(self, name):
        return self._scores[name]


def _stub_evaluate(monkeypatch, scores: dict[str, list[float]]):
    monkeypatch.setattr(run_ragas_eval_tool, "get_llm", lambda temperature=0: object())
    monkeypatch.setattr(run_ragas_eval_tool, "LangchainLLMWrapper", lambda llm: object())
    monkeypatch.setattr(run_ragas_eval_tool, "LangchainEmbeddingsWrapper", lambda embeddings: object())
    monkeypatch.setattr(run_ragas_eval_tool, "OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(
        run_ragas_eval_tool, "evaluate", lambda **kwargs: FakeRagasResult(scores)
    )


def _seed_compare_event(tmp_path, session_id="s1"):
    append_event(
        SessionEvent(
            session_id=session_id, turn_index=0, event="compare_resume_jd",
            resume_id="resume-01", job_id="job-01",
            user_input="Does the candidate match?",
            retrieved_contexts=["some evidence"],
            response="Strong match on core requirements.",
        ),
        sessions_dir=tmp_path,
    )


def test_run_ragas_eval_raises_when_no_eval_ready_events(tmp_path):
    with pytest.raises(ValueError, match="no eval-ready"):
        run_ragas_eval("missing-session", sessions_dir=tmp_path)


def test_run_ragas_eval_raises_for_reference_requiring_metric_without_reference(tmp_path, monkeypatch):
    _seed_compare_event(tmp_path)
    _stub_evaluate(monkeypatch, {"context_precision": [0.5]})

    with pytest.raises(ValueError, match="context_precision"):
        run_ragas_eval("s1", metric_names=["context_precision"], sessions_dir=tmp_path)


def test_run_ragas_eval_returns_aggregated_scores_for_default_metrics(tmp_path, monkeypatch):
    _seed_compare_event(tmp_path)
    _stub_evaluate(monkeypatch, {"faithfulness": [0.8, 1.0], "answer_relevancy": [0.6]})

    result = run_ragas_eval("s1", sessions_dir=tmp_path)

    assert result.session_id == "s1"
    assert result.sample_count == 1
    assert result.metric_names == ["faithfulness", "answer_relevancy"]
    assert result.scores["faithfulness"] == pytest.approx(0.9)
    assert result.scores["answer_relevancy"] == pytest.approx(0.6)
