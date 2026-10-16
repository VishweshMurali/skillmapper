from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

from decide_retrieval_tool import decide_retrieval as _decide_retrieval
from mcp.server.mcpserver import MCPServer

from skillmapper.ingestion.pipeline import bm25_path_for
from skillmapper.orchestration import session_log
from skillmapper.orchestration.schemas import DecideRetrievalResult
from skillmapper.retrieval import vector_store
from skillmapper.retrieval.bm25_index import BM25Index
from skillmapper.retrieval.embeddings import Embedder
from skillmapper.retrieval.hybrid import HybridRetriever
from skillmapper.retrieval.vector_store import METADATA_COLLECTION

mcp = MCPServer("rag-router")

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """Lazily built on first tool call -- loading Chroma + the BM25 index
    isn't free, and tool discovery (list_tools) shouldn't have to pay for it.
    """
    global _retriever
    if _retriever is None:
        logger.info("initializing HybridRetriever (Chroma + BM25)...")
        client = vector_store.get_client()
        collection = vector_store.get_collection(client, METADATA_COLLECTION)
        bm25 = BM25Index.load(bm25_path_for(METADATA_COLLECTION))
        _retriever = HybridRetriever(collection, bm25, Embedder())
    return _retriever


@mcp.tool()
def decide_retrieval(
    requirement: str, resume_id: str, session_id: str, turn_index: int, retry_count: int = 0
) -> DecideRetrievalResult:
    """Hybrid-retrieval routing for one requirement, widening k on retry."""
    result = _decide_retrieval(_get_retriever(), requirement, resume_id, retry_count=retry_count)
    session_log.append_event(
        session_log.SessionEvent(
            session_id=session_id,
            turn_index=turn_index,
            event="decide_retrieval",
            resume_id=resume_id,
            user_input=requirement,
            retrieved_contexts=[e.text for e in result.evidence],
            payload=result.model_dump(mode="json"),
        )
    )
    return result


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
