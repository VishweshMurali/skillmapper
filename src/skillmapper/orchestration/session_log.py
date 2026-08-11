from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from skillmapper.config import settings

_LOCK = threading.Lock()

EventType = Literal[
    "parse_jd", "decide_retrieval", "compare_resume_jd", "score_comparison", "session_summary"
]


class SessionEvent(BaseModel):
    session_id: str
    turn_index: int
    event: EventType
    resume_id: str | None = None
    job_id: str | None = None
    user_input: str | None = None
    retrieved_contexts: list[str] = Field(default_factory=list)
    response: str | None = None
    reference: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


def session_log_path(session_id: str, sessions_dir: Path | None = None) -> Path:
    base = sessions_dir if sessions_dir is not None else settings.sessions_dir
    return base / f"{session_id}.jsonl"


def append_event(event: SessionEvent, sessions_dir: Path | None = None) -> None:
    path = session_log_path(event.session_id, sessions_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = event.model_dump_json() + "\n"
    with _LOCK, path.open("a", encoding="utf-8") as f:
        f.write(line)


def read_session(session_id: str, sessions_dir: Path | None = None) -> list[SessionEvent]:
    path = session_log_path(session_id, sessions_dir)
    if not path.exists():
        return []
    return [
        SessionEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def to_ragas_rows(events: list[SessionEvent]) -> list[dict]:
    """Only `compare_resume_jd` events carry a complete (question, contexts,
    response) triple -- `decide_retrieval` has contexts but no response yet,
    `score_comparison` has neither.
    """
    return [
        {
            "user_input": e.user_input,
            "retrieved_contexts": e.retrieved_contexts,
            "response": e.response,
            "reference": e.reference,
        }
        for e in events
        if e.event == "compare_resume_jd" and e.user_input and e.response
    ]
