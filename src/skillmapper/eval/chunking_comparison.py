from __future__ import annotations

from skillmapper.eval.ragas_harness import run_ragas_eval
from skillmapper.eval.report import aggregate_scores, format_comparison_table
from skillmapper.eval.testset import TESTSET
from skillmapper.retrieval.vector_store import METADATA_COLLECTION, NAIVE_COLLECTION


def run_comparison() -> str:
    """Runs the identical testset + QA pipeline against the naive and
    metadata-aware collections, everything else held constant, and returns
    a markdown table -- the reproducible artifact behind the chunking
    debugging story: naive chunking crosses section boundaries, which
    RAGAS's faithfulness/context recall metrics should reflect as lower
    scores relative to metadata-aware chunking under the same local judge.
    """
    print(f"Running RAGAS against naive collection ({len(TESTSET)} questions)...")
    naive_result = run_ragas_eval(NAIVE_COLLECTION)
    naive_scores = aggregate_scores(naive_result)
    print("naive:", naive_scores)

    print(f"Running RAGAS against metadata-aware collection ({len(TESTSET)} questions)...")
    meta_result = run_ragas_eval(METADATA_COLLECTION)
    meta_scores = aggregate_scores(meta_result)
    print("metadata-aware:", meta_scores)

    return format_comparison_table(naive_scores, meta_scores)
