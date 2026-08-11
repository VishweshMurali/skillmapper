from skillmapper.orchestration.session_log import (
    SessionEvent,
    append_event,
    read_session,
    to_ragas_rows,
)


def test_append_and_read_session_round_trips(tmp_path):
    append_event(
        SessionEvent(
            session_id="s1", turn_index=0, event="parse_jd",
            job_id="job-01", payload={"title": "Backend Engineer"},
        ),
        sessions_dir=tmp_path,
    )
    append_event(
        SessionEvent(
            session_id="s1", turn_index=1, event="decide_retrieval",
            resume_id="resume-01", user_input="Kubernetes experience",
            retrieved_contexts=["Ran Kubernetes in production for 3 years."],
        ),
        sessions_dir=tmp_path,
    )

    events = read_session("s1", sessions_dir=tmp_path)
    assert len(events) == 2
    assert events[0].event == "parse_jd"
    assert events[0].payload == {"title": "Backend Engineer"}
    assert events[1].event == "decide_retrieval"
    assert events[1].retrieved_contexts == ["Ran Kubernetes in production for 3 years."]


def test_read_session_returns_empty_list_for_unknown_session(tmp_path):
    assert read_session("does-not-exist", sessions_dir=tmp_path) == []


def test_append_event_is_append_only_across_calls(tmp_path):
    for i in range(3):
        append_event(
            SessionEvent(session_id="s2", turn_index=i, event="score_comparison"),
            sessions_dir=tmp_path,
        )
    events = read_session("s2", sessions_dir=tmp_path)
    assert [e.turn_index for e in events] == [0, 1, 2]


def test_to_ragas_rows_only_includes_complete_compare_events(tmp_path):
    events = [
        SessionEvent(
            session_id="s3", turn_index=0, event="decide_retrieval",
            user_input="Kubernetes experience", retrieved_contexts=["some evidence"],
        ),
        SessionEvent(
            session_id="s3", turn_index=1, event="compare_resume_jd",
            resume_id="resume-01", job_id="job-01",
            user_input="Does Priya match the Backend Engineer role at Acme?",
            retrieved_contexts=["some evidence"],
            response="Strong match on core requirements.",
        ),
        SessionEvent(session_id="s3", turn_index=2, event="score_comparison"),
    ]

    rows = to_ragas_rows(events)

    assert len(rows) == 1
    assert rows[0]["user_input"] == "Does Priya match the Backend Engineer role at Acme?"
    assert rows[0]["response"] == "Strong match on core requirements."
    assert rows[0]["reference"] is None


def test_to_ragas_rows_excludes_compare_events_missing_response():
    events = [
        SessionEvent(
            session_id="s4", turn_index=0, event="compare_resume_jd",
            user_input="Does the candidate match?", response=None,
        )
    ]
    assert to_ragas_rows(events) == []
