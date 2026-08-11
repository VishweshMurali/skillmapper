from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from skillmapper.config import settings
from skillmapper.ingestion.chunking import chunk_document
from skillmapper.ingestion.loaders import load_all_job_descriptions, load_all_resumes
from skillmapper.models import Chunk, DocType
from skillmapper.retrieval import vector_store
from skillmapper.retrieval.bm25_index import BM25Index
from skillmapper.retrieval.embeddings import Embedder

ChunkerFn = Callable[[str, str, DocType], list[Chunk]]


def bm25_path_for(collection_name: str) -> Path:
    return settings.cache_dir / f"bm25_{collection_name}.pkl"


async def ingest(
    collection_name: str,
    chunker: ChunkerFn = chunk_document,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Load the synthetic corpus, chunk it with `chunker`, embed (cached,
    async-parallel) and upsert into the named Chroma collection, then
    rebuild that collection's BM25 index. Returns the chunks ingested.
    """
    embedder = embedder or Embedder()

    resumes = load_all_resumes(settings.synthetic_resumes_dir)
    jobs = load_all_job_descriptions(settings.synthetic_jobs_dir)

    all_chunks: list[Chunk] = []
    for resume in resumes:
        all_chunks.extend(chunker(resume.id, resume.raw_text, "resume"))
    for job in jobs:
        all_chunks.extend(chunker(job.id, job.raw_text, "job_description"))

    embeddings = await embedder.aembed_batch([c.text for c in all_chunks], "document")

    client = vector_store.get_client()
    collection = vector_store.get_collection(client, collection_name)
    vector_store.upsert_chunks(collection, all_chunks, embeddings)

    entries = vector_store.get_all_chunks(collection)
    bm25 = BM25Index.build_from_collection(entries)
    bm25.save(bm25_path_for(collection_name))

    return all_chunks


def ingest_sync(collection_name: str, chunker: ChunkerFn = chunk_document) -> list[Chunk]:
    return asyncio.run(ingest(collection_name, chunker))
