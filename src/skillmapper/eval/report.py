from __future__ import annotations

import math

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def aggregate_scores(result, metric_names: list[str] = METRIC_NAMES) -> dict[str, float]:
    """Mean of each metric's per-sample scores, ignoring NaN (a metric can
    legitimately NaN on a single row -- e.g. context_precision/recall are
    undefined when nothing relevant was retrieved -- without invalidating
    the rest of the run).
    """
    scores: dict[str, float] = {}
    for name in metric_names:
        values = [v for v in result[name] if v is not None and not math.isnan(v)]
        scores[name] = sum(values) / len(values) if values else float("nan")
    return scores


def format_comparison_table(
    naive_scores: dict[str, float], meta_scores: dict[str, float]
) -> str:
    lines = [
        "| Metric | Naive chunking | Metadata-aware chunking | Delta |",
        "|---|---|---|---|",
    ]
    for name in METRIC_NAMES:
        naive_v = naive_scores.get(name, float("nan"))
        meta_v = meta_scores.get(name, float("nan"))
        delta = meta_v - naive_v
        lines.append(f"| {name} | {naive_v:.3f} | {meta_v:.3f} | {delta:+.3f} |")
    return "\n".join(lines)


def format_single_table(scores: dict[str, float], title: str) -> str:
    lines = [f"### {title}", "", "| Metric | Score |", "|---|---|"]
    for name in METRIC_NAMES:
        lines.append(f"| {name} | {scores.get(name, float('nan')):.3f} |")
    return "\n".join(lines)
