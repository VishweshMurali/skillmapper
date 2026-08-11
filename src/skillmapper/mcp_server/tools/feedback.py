from __future__ import annotations

from skillmapper.mcp_server.schemas import CaptureFeedbackResult
from skillmapper.models import ReviewerFeedback
from skillmapper.storage import repository


def capture_reviewer_feedback(
    evaluation_id: str,
    reviewer_name: str,
    decision: str,
    corrected_verdict: str | None = None,
    comments: str | None = None,
) -> CaptureFeedbackResult:
    """Same underlying repository call the Streamlit reviewer app uses, so
    both entry points write feedback identically and can't drift.
    """
    if repository.get_evaluation(evaluation_id) is None:
        raise ValueError(f"unknown evaluation_id: {evaluation_id}")

    feedback = ReviewerFeedback(
        evaluation_id=evaluation_id,
        reviewer_name=reviewer_name,
        decision=decision,
        corrected_verdict=corrected_verdict,
        comments=comments,
    )
    repository.save_feedback(feedback)
    return CaptureFeedbackResult(feedback_id=feedback.id, status="saved")
