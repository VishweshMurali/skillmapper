import math

from skillmapper.eval.report import (
    aggregate_scores,
    format_comparison_table,
    format_single_table,
)


class FakeRagasResult:
    def __init__(self, data: dict[str, list[float]]):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


def test_aggregate_scores_computes_mean():
    result = FakeRagasResult(
        {
            "faithfulness": [1.0, 0.5],
            "answer_relevancy": [0.8, 0.6],
            "context_precision": [1.0, 1.0],
            "context_recall": [0.5, 0.5],
        }
    )
    scores = aggregate_scores(result)
    assert scores["faithfulness"] == 0.75
    assert scores["context_precision"] == 1.0


def test_aggregate_scores_ignores_nan():
    result = FakeRagasResult(
        {
            "faithfulness": [1.0, float("nan")],
            "answer_relevancy": [0.8],
            "context_precision": [float("nan")],
            "context_recall": [1.0, 0.0],
        }
    )
    scores = aggregate_scores(result)
    assert scores["faithfulness"] == 1.0
    assert math.isnan(scores["context_precision"])


def test_format_comparison_table_shows_delta():
    naive = {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5}
    meta = {"faithfulness": 0.9, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.8}
    table = format_comparison_table(naive, meta)
    assert "faithfulness" in table
    assert "+0.400" in table
    assert "+0.300" in table


def test_format_single_table_includes_all_metrics():
    scores = {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.6}
    table = format_single_table(scores, "My Title")
    assert "My Title" in table
    for name in scores:
        assert name in table
