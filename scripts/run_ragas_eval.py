"""Run RAGAS (faithfulness, answer_relevancy, context_precision, context_recall)
against the metadata-aware (production) collection, judged by the local
Ollama model. With ~12 questions x 4 metrics on a local judge, expect this
to take several minutes.

Usage:
    uv run scripts/run_ragas_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmapper.eval.ragas_harness import run_ragas_eval
from skillmapper.eval.report import aggregate_scores, format_single_table
from skillmapper.retrieval.vector_store import METADATA_COLLECTION


def main() -> None:
    result = run_ragas_eval(METADATA_COLLECTION)
    scores = aggregate_scores(result)
    print()
    print(format_single_table(scores, "RAGAS: metadata-aware chunking"))


if __name__ == "__main__":
    main()
