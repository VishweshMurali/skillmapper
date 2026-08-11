from __future__ import annotations

# Imported, not duplicated: this repo's 6 new workspace members share one uv
# workspace venv, so importing costs nothing extra and keeps one source of
# truth for the evidence-coverage threshold the agent's `retrieve` node and
# this scorer must agree on.
from skillmapper.agent.nodes import MIN_EVIDENCE_SCORE
from skillmapper.models import EvidenceItem
from skillmapper.orchestration.schemas import ScoreResult


def score_comparison(evidence: list[EvidenceItem], self_reported_confidence: float) -> ScoreResult:
    """Evidence-coverage-dominant confidence, ported verbatim from
    agent/nodes.py:confidence_check -- an 8B model's self-reported confidence
    is close to uncalibrated noise, so it's kept but weighted lightly.
    """
    by_requirement: dict[str, list[EvidenceItem]] = {}
    for item in evidence:
        by_requirement.setdefault(item.supports_requirement, []).append(item)

    if by_requirement:
        covered = sum(
            1
            for items in by_requirement.values()
            if any(i.score >= MIN_EVIDENCE_SCORE for i in items)
        )
        evidence_coverage = covered / len(by_requirement)
    else:
        evidence_coverage = 0.0

    confidence = 0.3 * self_reported_confidence + 0.7 * evidence_coverage
    return ScoreResult(
        confidence=confidence,
        evidence_coverage=evidence_coverage,
        self_reported_confidence=self_reported_confidence,
    )
