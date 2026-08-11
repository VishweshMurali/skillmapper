from __future__ import annotations

from pathlib import Path

from skillmapper.orchestration import session_log
from skillmapper.orchestration.schemas import EvalLogsResource


def eval_logs(session_id: str, sessions_dir: Path | None = None) -> EvalLogsResource:
    events = session_log.read_session(session_id, sessions_dir)
    return EvalLogsResource(
        session_id=session_id,
        event_count=len(events),
        events=[e.model_dump(mode="json") for e in events],
    )
