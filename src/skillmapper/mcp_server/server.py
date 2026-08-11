from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")

# stdio transport requires stdout stay clean for the JSON-RPC channel --
# all logging goes to stderr, never print().
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

from mcp.server.mcpserver import MCPServer

from skillmapper.agent.runtime import AgentRuntime
from skillmapper.mcp_server.schemas import (
    CaptureFeedbackResult,
    LookupEvaluationsResult,
    RetrieveEvidenceResult,
    SearchDocumentsResult,
)
from skillmapper.mcp_server.tools.compare import (
    compare_resume_to_job as _compare_resume_to_job,
)
from skillmapper.mcp_server.tools.feedback import (
    capture_reviewer_feedback as _capture_reviewer_feedback,
)
from skillmapper.mcp_server.tools.lookup import (
    lookup_evaluation_result as _lookup_evaluation_result,
)
from skillmapper.mcp_server.tools.retrieve_evidence import (
    retrieve_evidence as _retrieve_evidence,
)
from skillmapper.mcp_server.tools.search_documents import (
    search_documents as _search_documents,
)
from skillmapper.models import EvaluationResult

mcp = MCPServer("skillmapper")

_runtime: AgentRuntime | None = None


def _get_runtime() -> AgentRuntime:
    """Lazily built on first tool call that needs it -- loading Chroma, the
    BM25 index, and the Ollama LLM client isn't free, and tool discovery
    (list_tools) shouldn't have to pay for it.
    """
    global _runtime
    if _runtime is None:
        logger.info("initializing AgentRuntime (Chroma + BM25 + Ollama)...")
        _runtime = AgentRuntime()
    return _runtime


@mcp.tool()
def search_documents(
    query: str, doc_type: str = "any", section: str | None = None, top_k: int = 5
) -> SearchDocumentsResult:
    """Hybrid (BM25 + dense) search over ingested resume/job-description chunks."""
    return _search_documents(
        _get_runtime().retriever, query, doc_type=doc_type, section=section, top_k=top_k
    )


@mcp.tool()
def retrieve_evidence(
    resume_id: str, job_id: str, requirement: str, top_k: int = 5
) -> RetrieveEvidenceResult:
    """Retrieve resume evidence supporting a specific job requirement."""
    return _retrieve_evidence(_get_runtime().retriever, resume_id, job_id, requirement, top_k=top_k)


@mcp.tool()
def compare_resume_to_job(
    resume_id: str, job_id: str, force_rerun: bool = False
) -> EvaluationResult:
    """Run the full resume-vs-job comparison agent (retrieval + LLM compare +
    retry/escalation). Expect 30s-several minutes on a local 8B model;
    returns a cached completed result unless force_rerun=True.
    """
    return _compare_resume_to_job(_get_runtime(), resume_id, job_id, force_rerun=force_rerun)


@mcp.tool()
def capture_reviewer_feedback(
    evaluation_id: str,
    reviewer_name: str,
    decision: str,
    corrected_verdict: str | None = None,
    comments: str | None = None,
) -> CaptureFeedbackResult:
    """Record a human reviewer's decision on an evaluation."""
    return _capture_reviewer_feedback(
        evaluation_id, reviewer_name, decision,
        corrected_verdict=corrected_verdict, comments=comments,
    )


@mcp.tool()
def lookup_evaluation_result(
    evaluation_id: str | None = None,
    resume_id: str | None = None,
    job_id: str | None = None,
    status: str | None = None,
) -> LookupEvaluationsResult:
    """Look up stored evaluation results by id, or filter by resume/job/status."""
    return _lookup_evaluation_result(
        evaluation_id=evaluation_id, resume_id=resume_id, job_id=job_id, status=status
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
