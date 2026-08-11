from skillmapper_orchestrator import retry_loop
from skillmapper_orchestrator.retry_loop import run_session

REQUIREMENT_BLOCKS = [{"section": "requirements", "heading": "Requirements", "text": "Must know Python"}]


class FakeClients:
    """Scripts each tool's response by name; `score_comparison`'s confidence
    is driven by a per-attempt sequence, modeling a retry that legitimately
    finds better evidence -- direct analog of test_agent_graph.py's
    ScoreByWidthRetriever.
    """

    def __init__(self, confidence_sequence: list[float]):
        self.confidence_sequence = confidence_sequence
        self.calls: list[tuple[str, str, dict]] = []
        self._attempt = 0

    async def call_tool(self, server_name, tool_name, arguments, timeout=None):
        self.calls.append((server_name, tool_name, dict(arguments)))

        if tool_name == "parse_jd":
            return {"job_id": arguments["job_id"], "title": "Engineer", "company": "Acme",
                     "sections": REQUIREMENT_BLOCKS, "requirement_blocks": REQUIREMENT_BLOCKS}
        if tool_name == "decide_retrieval":
            return {
                "requirement": arguments["requirement"], "k_used": 5,
                "evidence": [{
                    "chunk_id": "r1::skills::0", "doc_id": arguments["resume_id"], "section": "skills",
                    "text": "Python expert", "score": 0.03, "supports_requirement": arguments["requirement"],
                }],
            }
        if tool_name == "compare_resume_jd":
            return {
                "resume_id": arguments["resume_id"], "job_id": arguments["job_id"],
                "verdict": "partial_match", "overall_score": 0.5, "matched_skills": ["Python"],
                "missing_skills": [], "reasoning": "stub reasoning", "self_reported_confidence": 0.5,
            }
        if tool_name == "score_comparison":
            confidence = self.confidence_sequence[self._attempt]
            self._attempt += 1
            return {"confidence": confidence, "evidence_coverage": confidence, "self_reported_confidence": 0.5}

        raise AssertionError(f"unexpected tool call: {server_name}.{tool_name}")


def _stub_repository(monkeypatch):
    saved = []
    monkeypatch.setattr(retry_loop.repository, "save_evaluation", lambda result: saved.append(result))
    return saved


async def test_retry_widens_k_and_then_finalizes(monkeypatch):
    saved = _stub_repository(monkeypatch)
    clients = FakeClients(confidence_sequence=[0.3, 0.8])

    session_id, result = await run_session(
        clients, "r1", "j1", threshold=0.65, max_retries=2
    )

    assert result.status == "completed"
    assert result.retries_used == 1
    retry_counts = [c[2]["retry_count"] for c in clients.calls if c[1] == "decide_retrieval"]
    assert retry_counts == [0, 1]
    assert len(saved) == 1
    assert saved[0].status == "completed"
    assert saved[0].retries_used == 1
    assert session_id


async def test_escalates_after_max_retries_when_confidence_never_recovers(monkeypatch):
    saved = _stub_repository(monkeypatch)
    clients = FakeClients(confidence_sequence=[0.1, 0.1, 0.1])

    session_id, result = await run_session(
        clients, "r1", "j1", threshold=0.65, max_retries=2
    )

    assert result.status == "escalated"
    assert result.retries_used == 2
    assert result.escalation_reason is not None
    retry_counts = [c[2]["retry_count"] for c in clients.calls if c[1] == "decide_retrieval"]
    assert retry_counts == [0, 1, 2]
    assert len(saved) == 1
    assert saved[0].status == "escalated"
