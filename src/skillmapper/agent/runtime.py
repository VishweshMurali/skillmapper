from __future__ import annotations

from skillmapper.agent.graph import build_graph, invoke_graph
from skillmapper.agent.llm import get_llm
from skillmapper.agent.state import make_initial_state
from skillmapper.config import settings
from skillmapper.ingestion.loaders import load_all_job_descriptions, load_all_resumes
from skillmapper.ingestion.pipeline import bm25_path_for
from skillmapper.models import EvaluationResult
from skillmapper.retrieval import vector_store
from skillmapper.retrieval.bm25_index import BM25Index
from skillmapper.retrieval.embeddings import Embedder
from skillmapper.retrieval.hybrid import HybridRetriever
from skillmapper.retrieval.vector_store import METADATA_COLLECTION
from skillmapper.storage import repository


class AgentRuntime:
    """Wires up the real retriever/LLM/corpus and exposes the one operation
    the MCP `compare_resume_to_job` tool and `scripts/run_agent.py` both
    need. Built once per process -- Ollama model loading and Chroma/BM25
    index loading are not free.
    """

    def __init__(self):
        client = vector_store.get_client()
        collection = vector_store.get_collection(client, METADATA_COLLECTION)
        bm25 = BM25Index.load(bm25_path_for(METADATA_COLLECTION))
        embedder = Embedder()
        self.retriever = HybridRetriever(collection, bm25, embedder)
        self.llm = get_llm()
        self.resumes_by_id = {r.id: r for r in load_all_resumes(settings.synthetic_resumes_dir)}
        self.jobs_by_id = {j.id: j for j in load_all_job_descriptions(settings.synthetic_jobs_dir)}
        self.graph = build_graph(self.retriever, self.llm, self.resumes_by_id, self.jobs_by_id)

    def compare_resume_to_job(
        self, resume_id: str, job_id: str, force_rerun: bool = False
    ) -> EvaluationResult:
        if resume_id not in self.resumes_by_id:
            raise ValueError(f"unknown resume_id: {resume_id}")
        if job_id not in self.jobs_by_id:
            raise ValueError(f"unknown job_id: {job_id}")

        if not force_rerun:
            existing = repository.find_completed_evaluation(resume_id, job_id)
            if existing is not None:
                return existing

        initial_state = make_initial_state(resume_id, job_id, settings.agent_max_retries)
        final_state = invoke_graph(self.graph, initial_state)
        return final_state["final_result"]
