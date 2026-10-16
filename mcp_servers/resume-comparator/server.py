from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

from compare_resume_jd_tool import compare_resume_jd as _compare_resume_jd
from mcp.server.mcpserver import MCPServer

from skillmapper.agent.llm import ChatOllama, get_llm
from skillmapper.config import settings
from skillmapper.ingestion.loaders import load_all_job_descriptions, load_all_resumes
from skillmapper.models import EvidenceItem, JobDescription, Resume
from skillmapper.orchestration import session_log
from skillmapper.orchestration.schemas import CompareResult

mcp = MCPServer("resume-comparator")

_llm: ChatOllama | None = None
_resumes_by_id: dict[str, Resume] | None = None
_jobs_by_id: dict[str, JobDescription] | None = None


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def _get_corpus() -> tuple[dict[str, Resume], dict[str, JobDescription]]:
    """Lazily loads the resume/job corpus once -- mirrors AgentRuntime.__init__."""
    global _resumes_by_id, _jobs_by_id
    if _resumes_by_id is None:
        logger.info("loading resume/job corpus...")
        _resumes_by_id = {r.id: r for r in load_all_resumes(settings.synthetic_resumes_dir)}
        _jobs_by_id = {j.id: j for j in load_all_job_descriptions(settings.synthetic_jobs_dir)}
    return _resumes_by_id, _jobs_by_id


@mcp.tool()
def compare_resume_jd(
    resume_id: str,
    job_id: str,
    evidence: list[EvidenceItem],
    session_id: str,
    turn_index: int,
) -> CompareResult:
    """LLM-judged resume-vs-job comparison over the given retrieved evidence."""
    resumes_by_id, jobs_by_id = _get_corpus()
    result = _compare_resume_jd(
        _get_llm(), resumes_by_id, jobs_by_id, resume_id, job_id, evidence
    )

    resume = resumes_by_id[resume_id]
    job = jobs_by_id[job_id]
    session_log.append_event(
        session_log.SessionEvent(
            session_id=session_id,
            turn_index=turn_index,
            event="compare_resume_jd",
            resume_id=resume_id,
            job_id=job_id,
            user_input=f"Does {resume.candidate_name} match the {job.title} role at {job.company}?",
            retrieved_contexts=[e.text for e in evidence],
            response=result.reasoning,
            payload=result.model_dump(mode="json"),
        )
    )
    return result


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
