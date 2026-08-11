from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from skillmapper.models import EvaluationResult, EvidenceItem, ReviewerFeedback
from skillmapper.storage.db import get_connection, init_db


def _row_to_evaluation(row: sqlite3.Row) -> EvaluationResult:
    return EvaluationResult(
        id=row["id"],
        resume_id=row["resume_id"],
        job_id=row["job_id"],
        verdict=row["verdict"],
        overall_score=row["overall_score"],
        matched_skills=json.loads(row["matched_skills"]),
        missing_skills=json.loads(row["missing_skills"]),
        evidence=[EvidenceItem(**e) for e in json.loads(row["evidence"])],
        reasoning=row["reasoning"],
        confidence=row["confidence"],
        status=row["status"],
        escalation_reason=row["escalation_reason"],
        retries_used=row["retries_used"],
        created_at=row["created_at"],
    )


def _row_to_feedback(row: sqlite3.Row) -> ReviewerFeedback:
    return ReviewerFeedback(
        id=row["id"],
        evaluation_id=row["evaluation_id"],
        reviewer_name=row["reviewer_name"],
        decision=row["decision"],
        corrected_verdict=row["corrected_verdict"],
        comments=row["comments"],
        created_at=row["created_at"],
    )


def save_evaluation(result: EvaluationResult, db_path: Path | None = None) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO evaluations
                (id, resume_id, job_id, verdict, overall_score, matched_skills,
                 missing_skills, evidence, reasoning, confidence, status,
                 escalation_reason, retries_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                verdict=excluded.verdict, overall_score=excluded.overall_score,
                matched_skills=excluded.matched_skills, missing_skills=excluded.missing_skills,
                evidence=excluded.evidence, reasoning=excluded.reasoning,
                confidence=excluded.confidence, status=excluded.status,
                escalation_reason=excluded.escalation_reason, retries_used=excluded.retries_used
            """,
            (
                result.id,
                result.resume_id,
                result.job_id,
                result.verdict,
                result.overall_score,
                json.dumps(result.matched_skills),
                json.dumps(result.missing_skills),
                json.dumps([e.model_dump() for e in result.evidence]),
                result.reasoning,
                result.confidence,
                result.status,
                result.escalation_reason,
                result.retries_used,
                result.created_at.isoformat(),
            ),
        )


def get_evaluation(evaluation_id: str, db_path: Path | None = None) -> EvaluationResult | None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
        ).fetchone()
        return _row_to_evaluation(row) if row else None


def find_completed_evaluation(
    resume_id: str, job_id: str, db_path: Path | None = None
) -> EvaluationResult | None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM evaluations
            WHERE resume_id = ? AND job_id = ? AND status IN ('completed', 'completed_reviewed')
            ORDER BY created_at DESC LIMIT 1
            """,
            (resume_id, job_id),
        ).fetchone()
        return _row_to_evaluation(row) if row else None


def query_evaluations(
    resume_id: str | None = None,
    job_id: str | None = None,
    status: str | None = None,
    db_path: Path | None = None,
) -> list[EvaluationResult]:
    init_db(db_path)
    clauses, params = [], []
    if resume_id is not None:
        clauses.append("resume_id = ?")
        params.append(resume_id)
    if job_id is not None:
        clauses.append("job_id = ?")
        params.append(job_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM evaluations {where} ORDER BY created_at DESC", params
        ).fetchall()
        return [_row_to_evaluation(r) for r in rows]


def save_feedback(feedback: ReviewerFeedback, db_path: Path | None = None) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO feedback
                (id, evaluation_id, reviewer_name, decision, corrected_verdict, comments, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback.id,
                feedback.evaluation_id,
                feedback.reviewer_name,
                feedback.decision,
                feedback.corrected_verdict,
                feedback.comments,
                feedback.created_at.isoformat(),
            ),
        )
        conn.execute(
            "UPDATE evaluations SET status = 'completed_reviewed' WHERE id = ?",
            (feedback.evaluation_id,),
        )


def list_feedback_for_evaluation(
    evaluation_id: str, db_path: Path | None = None
) -> list[ReviewerFeedback]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE evaluation_id = ? ORDER BY created_at DESC",
            (evaluation_id,),
        ).fetchall()
        return [_row_to_feedback(r) for r in rows]
