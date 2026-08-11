from skillmapper.models import EvaluationResult, EvidenceItem, ReviewerFeedback
from skillmapper.storage import repository


def _sample_result(**overrides) -> EvaluationResult:
    defaults = dict(
        resume_id="resume-01",
        job_id="job-01",
        verdict="strong_match",
        overall_score=0.82,
        matched_skills=["Python", "Kubernetes"],
        missing_skills=["Go"],
        evidence=[
            EvidenceItem(
                chunk_id="resume-01::skills::0",
                doc_id="resume-01",
                section="skills",
                text="Python, Go, PostgreSQL",
                score=0.9,
                supports_requirement="Strong proficiency in Python or Go",
            )
        ],
        reasoning="Strong overlap on core requirements.",
        confidence=0.78,
        status="completed",
        retries_used=0,
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def test_save_and_get_evaluation_round_trips(tmp_path):
    db_path = tmp_path / "test.db"
    result = _sample_result()
    repository.save_evaluation(result, db_path=db_path)

    fetched = repository.get_evaluation(result.id, db_path=db_path)
    assert fetched is not None
    assert fetched.id == result.id
    assert fetched.matched_skills == ["Python", "Kubernetes"]
    assert fetched.evidence[0].chunk_id == "resume-01::skills::0"
    assert fetched.status == "completed"


def test_query_evaluations_filters_by_status(tmp_path):
    db_path = tmp_path / "test.db"
    repository.save_evaluation(_sample_result(status="completed"), db_path=db_path)
    repository.save_evaluation(
        _sample_result(status="escalated", escalation_reason="low confidence"), db_path=db_path
    )

    escalated = repository.query_evaluations(status="escalated", db_path=db_path)
    assert len(escalated) == 1
    assert escalated[0].escalation_reason == "low confidence"

    all_results = repository.query_evaluations(db_path=db_path)
    assert len(all_results) == 2


def test_find_completed_evaluation_ignores_escalated(tmp_path):
    db_path = tmp_path / "test.db"
    repository.save_evaluation(
        _sample_result(resume_id="r1", job_id="j1", status="escalated"), db_path=db_path
    )
    assert repository.find_completed_evaluation("r1", "j1", db_path=db_path) is None

    repository.save_evaluation(
        _sample_result(resume_id="r1", job_id="j1", status="completed"), db_path=db_path
    )
    found = repository.find_completed_evaluation("r1", "j1", db_path=db_path)
    assert found is not None
    assert found.status == "completed"


def test_save_feedback_marks_evaluation_reviewed(tmp_path):
    db_path = tmp_path / "test.db"
    result = _sample_result(status="escalated")
    repository.save_evaluation(result, db_path=db_path)

    feedback = ReviewerFeedback(
        evaluation_id=result.id,
        reviewer_name="Alex",
        decision="approve",
    )
    repository.save_feedback(feedback, db_path=db_path)

    updated = repository.get_evaluation(result.id, db_path=db_path)
    assert updated.status == "completed_reviewed"

    stored_feedback = repository.list_feedback_for_evaluation(result.id, db_path=db_path)
    assert len(stored_feedback) == 1
    assert stored_feedback[0].reviewer_name == "Alex"
