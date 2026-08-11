from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from skillmapper.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    resume_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    overall_score REAL NOT NULL,
    matched_skills TEXT NOT NULL,
    missing_skills TEXT NOT NULL,
    evidence TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    escalation_reason TEXT,
    retries_used INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
    reviewer_name TEXT NOT NULL,
    decision TEXT NOT NULL,
    corrected_verdict TEXT,
    comments TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evaluations_status ON evaluations(status);
CREATE INDEX IF NOT EXISTS idx_evaluations_resume_job ON evaluations(resume_id, job_id);
CREATE INDEX IF NOT EXISTS idx_feedback_evaluation ON feedback(evaluation_id);
"""


def init_db(path: Path | None = None) -> None:
    path = path or settings.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(_SCHEMA)


@contextmanager
def get_connection(path: Path | None = None):
    """Short-lived per-call connection -- Streamlit and the MCP server are
    separate processes that may write concurrently on Windows; WAL mode +
    not holding connections open avoids 'database is locked' errors.
    """
    path = path or settings.sqlite_path
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
