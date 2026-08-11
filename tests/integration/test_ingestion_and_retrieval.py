"""Requires a running local Ollama with llama3.1:8b + nomic-embed-text
pulled, and `scripts/ingest.py` having been run at least once.

Run with: uv run pytest -m ollama tests/integration
"""

import pytest

from skillmapper.ingestion.pipeline import bm25_path_for
from skillmapper.retrieval import vector_store
from skillmapper.retrieval.bm25_index import BM25Index
from skillmapper.retrieval.embeddings import Embedder
from skillmapper.retrieval.hybrid import HybridRetriever
from skillmapper.retrieval.vector_store import METADATA_COLLECTION

pytestmark = [pytest.mark.ollama, pytest.mark.slow]


@pytest.fixture
def retriever() -> HybridRetriever:
    client = vector_store.get_client()
    collection = vector_store.get_collection(client, METADATA_COLLECTION)
    bm25 = BM25Index.load(bm25_path_for(METADATA_COLLECTION))
    return HybridRetriever(collection, bm25, Embedder())


def test_metadata_aware_collection_is_populated(retriever):
    entries = vector_store.get_all_chunks(retriever.collection)
    assert len(entries) > 0
    assert all(e["metadata"]["section"] for e in entries)


def test_exact_certification_keyword_surfaces_via_bm25(retriever):
    # "CKA" is an exact acronym that only a keyword match (not pure
    # semantic similarity) reliably surfaces -- this is the point of the
    # hybrid retriever.
    results = retriever.search("CKA certification", k=5, doc_type="resume")
    assert any("CKA" in r["text"] for r in results)


def test_search_scoped_to_one_resume_excludes_others(retriever):
    results = retriever.search("Python", k=10, doc_id="resume-01-senior-swe-backend")
    assert results
    assert all(r["metadata"]["doc_id"] == "resume-01-senior-swe-backend" for r in results)
