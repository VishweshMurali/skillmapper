"""Confirms the RAGAS harness wiring (local Ollama judge, dataset
construction, metric imports) actually executes -- not a real evaluation
run. Run the full 4-metric x naive/meta comparison via
scripts/run_chunking_comparison.py, not here (it takes tens of minutes).

Run with: uv run pytest -m ollama tests/integration
"""

import math

import pytest

from skillmapper.eval.ragas_harness import run_ragas_eval
from skillmapper.eval.report import aggregate_scores
from skillmapper.eval.testset import TESTSET
from skillmapper.retrieval.vector_store import METADATA_COLLECTION

pytestmark = [pytest.mark.ollama, pytest.mark.slow]


def test_ragas_harness_runs_on_a_small_subset():
    result = run_ragas_eval(METADATA_COLLECTION, testset=TESTSET[:2])
    scores = aggregate_scores(result)
    assert set(scores) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    for value in scores.values():
        assert math.isnan(value) or 0.0 <= value <= 1.0  # allow NaN, else must be in [0,1]
