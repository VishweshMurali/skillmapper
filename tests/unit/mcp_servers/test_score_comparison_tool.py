from score_comparison_tool import score_comparison

from skillmapper.models import EvidenceItem


def _evidence(requirement: str, score: float) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=f"resume-01::skills::{requirement}",
        doc_id="resume-01",
        section="skills",
        text="some evidence text",
        score=score,
        supports_requirement=requirement,
    )


def test_full_coverage_and_high_self_report_yields_high_confidence():
    evidence = [_evidence("Python", 0.03), _evidence("Kubernetes", 0.03)]

    result = score_comparison(evidence, self_reported_confidence=0.9)

    assert result.evidence_coverage == 1.0
    assert result.confidence == 0.3 * 0.9 + 0.7 * 1.0


def test_partial_coverage_when_one_requirement_has_no_evidence_above_threshold():
    evidence = [_evidence("Python", 0.03), _evidence("Kubernetes", 0.0)]

    result = score_comparison(evidence, self_reported_confidence=0.8)

    assert result.evidence_coverage == 0.5
    assert result.confidence == 0.3 * 0.8 + 0.7 * 0.5


def test_no_evidence_yields_zero_coverage_regardless_of_self_report():
    result = score_comparison([], self_reported_confidence=0.95)

    assert result.evidence_coverage == 0.0
    assert result.confidence == 0.3 * 0.95
    assert result.self_reported_confidence == 0.95


def test_multiple_evidence_items_for_same_requirement_count_once():
    evidence = [_evidence("Python", 0.03), _evidence("Python", 0.0)]

    result = score_comparison(evidence, self_reported_confidence=0.5)

    assert result.evidence_coverage == 1.0
