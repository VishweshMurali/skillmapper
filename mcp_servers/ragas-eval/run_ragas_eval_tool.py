from __future__ import annotations

import warnings
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

with warnings.catch_warnings():
    # classic ragas.metrics import path is deprecated in favor of
    # ragas.metrics.collections as of 0.4.x, but still supported and is
    # what evaluate() expects here -- mirrors eval/ragas_harness.py.
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

from skillmapper.agent.llm import get_llm
from skillmapper.config import settings
from skillmapper.eval.report import aggregate_scores
from skillmapper.orchestration import session_log
from skillmapper.orchestration.schemas import RagasEvalResult

# Only the two reference-free metrics by default -- live orchestrator
# sessions have no ground-truth `reference` (see REFERENCE_REQUIRING_METRICS
# below), unlike the static testset eval/ragas_harness.py scores.
DEFAULT_METRICS = ["faithfulness", "answer_relevancy"]
REFERENCE_REQUIRING_METRICS = {"context_precision", "context_recall"}

_METRIC_REGISTRY = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
}


def run_ragas_eval(
    session_id: str, metric_names: list[str] | None = None, sessions_dir: Path | None = None
) -> RagasEvalResult:
    """Scores one orchestrator session's logged compare_resume_jd turns --
    small N, seconds not minutes, so this stays well clear of MCP client
    timeouts (unlike the 20-60+ min full-testset eval/ragas_harness.py run).
    """
    events = session_log.read_session(session_id, sessions_dir)
    rows = session_log.to_ragas_rows(events)
    if not rows:
        raise ValueError(f"no eval-ready compare_resume_jd events found for session {session_id!r}")

    names = metric_names or DEFAULT_METRICS
    requested_reference_metrics = REFERENCE_REQUIRING_METRICS & set(names)
    if requested_reference_metrics and not any(row["reference"] for row in rows):
        raise ValueError(
            f"{sorted(requested_reference_metrics)} require a `reference` value per row; "
            f"session {session_id!r} has none (no ground truth for live comparisons)"
        )

    dataset = EvaluationDataset(samples=[SingleTurnSample(**row) for row in rows])
    judge_llm = LangchainLLMWrapper(get_llm(temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        metric_objs = [_METRIC_REGISTRY[name]() for name in names]

    result = evaluate(
        dataset=dataset,
        metrics=metric_objs,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=2, timeout=180),
        raise_exceptions=False,
    )
    scores = aggregate_scores(result, names)
    return RagasEvalResult(session_id=session_id, sample_count=len(rows), scores=scores, metric_names=names)
