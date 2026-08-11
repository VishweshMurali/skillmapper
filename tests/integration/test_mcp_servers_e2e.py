"""End-to-end smoke tests for the 5 standalone MCP servers under mcp_servers/.

Each test spawns the real server as a subprocess over stdio and calls its
tool through a real mcp ClientSession -- this is what validates the
@mcp.tool() + Pydantic-return-type wiring actually round-trips through the
MCP wire protocol, which a fake-collaborator unit test cannot check.

jd-parser and scorer need neither Ollama nor a populated vector index, so
their tests run by default. rag-router/resume-comparator/ragas-eval are
gated behind the `requires_index`/`ollama`/`slow` markers already used
elsewhere in this suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from skillmapper.orchestration.session_log import SessionEvent, append_event, session_log_path

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_SERVERS_DIR = REPO_ROOT / "mcp_servers"


async def test_jd_parser_server_parse_jd_round_trips():
    session_id = "smoke-test-jd-parser"
    try:
        params = StdioServerParameters(
            command=sys.executable, args=[str(MCP_SERVERS_DIR / "jd-parser" / "server.py")]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "parse_jd",
                    {"job_id": "job-01-swe-backend", "session_id": session_id, "turn_index": 0},
                )
                assert result.is_error is False
                data = result.structured_content
                assert data["job_id"] == "job-01-swe-backend"
                assert data["title"]
                assert any(
                    b["section"] in {"requirements", "qualifications"}
                    for b in data["requirement_blocks"]
                )
    finally:
        session_log_path(session_id).unlink(missing_ok=True)


async def test_scorer_server_score_comparison_round_trips():
    session_id = "smoke-test-scorer"
    evidence = [
        {
            "chunk_id": "resume-01::skills::0",
            "doc_id": "resume-01",
            "section": "skills",
            "text": "5 years of Python and Kubernetes.",
            "score": 0.03,
            "supports_requirement": "Python",
        }
    ]
    try:
        params = StdioServerParameters(
            command=sys.executable, args=[str(MCP_SERVERS_DIR / "scorer" / "server.py")]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "score_comparison",
                    {
                        "evidence": evidence,
                        "self_reported_confidence": 0.9,
                        "session_id": session_id,
                        "turn_index": 0,
                    },
                )
                assert result.is_error is False
                data = result.structured_content
                assert data["evidence_coverage"] == 1.0
                assert data["confidence"] == 0.3 * 0.9 + 0.7 * 1.0
    finally:
        session_log_path(session_id).unlink(missing_ok=True)


@pytest.mark.requires_index
async def test_rag_router_server_decide_retrieval_round_trips():
    session_id = "smoke-test-rag-router"
    try:
        params = StdioServerParameters(
            command=sys.executable, args=[str(MCP_SERVERS_DIR / "rag-router" / "server.py")]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "decide_retrieval",
                    {
                        "requirement": "Strong proficiency in Python or Go",
                        "resume_id": "resume-01-senior-swe-backend",
                        "session_id": session_id,
                        "turn_index": 0,
                        "retry_count": 0,
                    },
                )
                assert result.is_error is False
                data = result.structured_content
                assert data["k_used"] == 5
                assert isinstance(data["evidence"], list)
                assert all(e["doc_id"] == "resume-01-senior-swe-backend" for e in data["evidence"])
    finally:
        session_log_path(session_id).unlink(missing_ok=True)


@pytest.mark.ollama
@pytest.mark.slow
async def test_resume_comparator_server_compare_resume_jd_round_trips():
    session_id = "smoke-test-resume-comparator"
    evidence = [
        {
            "chunk_id": "resume-01-senior-swe-backend::skills::0",
            "doc_id": "resume-01-senior-swe-backend",
            "section": "skills",
            "text": "5 years of Python, Go, Kubernetes, and PostgreSQL.",
            "score": 0.03,
            "supports_requirement": "Strong proficiency in Python or Go",
        }
    ]
    try:
        params = StdioServerParameters(
            command=sys.executable, args=[str(MCP_SERVERS_DIR / "resume-comparator" / "server.py")]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "compare_resume_jd",
                    {
                        "resume_id": "resume-01-senior-swe-backend",
                        "job_id": "job-01-swe-backend",
                        "evidence": evidence,
                        "session_id": session_id,
                        "turn_index": 0,
                    },
                    read_timeout_seconds=600,
                )
                assert result.is_error is False
                data = result.structured_content
                assert data["verdict"] in {"strong_match", "partial_match", "weak_match"}
                assert 0.0 <= data["overall_score"] <= 1.0
                assert data["reasoning"]
    finally:
        session_log_path(session_id).unlink(missing_ok=True)


@pytest.mark.ollama
@pytest.mark.slow
async def test_ragas_eval_server_run_ragas_eval_and_eval_logs_round_trip():
    session_id = "smoke-test-ragas-eval"
    append_event(
        SessionEvent(
            session_id=session_id, turn_index=0, event="compare_resume_jd",
            resume_id="resume-01-senior-swe-backend", job_id="job-01-swe-backend",
            user_input="Does Priya Natarajan match the Senior Backend Software Engineer role at Northwind Payments?",
            retrieved_contexts=["5 years of Python, Go, Kubernetes, and PostgreSQL."],
            response="Strong overlap: Python, Go, Kubernetes, and PostgreSQL all match core requirements.",
        )
    )
    try:
        params = StdioServerParameters(
            command=sys.executable, args=[str(MCP_SERVERS_DIR / "ragas-eval" / "server.py")]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tool_result = await session.call_tool(
                    "run_ragas_eval", {"session_id": session_id}, read_timeout_seconds=180
                )
                assert tool_result.is_error is False
                data = tool_result.structured_content
                assert data["session_id"] == session_id
                assert data["sample_count"] == 1
                assert set(data["metric_names"]) == {"faithfulness", "answer_relevancy"}
                assert all(0.0 <= v <= 1.0 for v in data["scores"].values())

                resource_result = await session.read_resource(f"session://{session_id}/events")
                logs = json.loads(resource_result.contents[0].text)
                assert logs["session_id"] == session_id
                assert logs["event_count"] == 1
    finally:
        session_log_path(session_id).unlink(missing_ok=True)
