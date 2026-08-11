from __future__ import annotations

import asyncio
import time
import uuid

from skillmapper.config import settings
from skillmapper.retrieval import vector_store
from skillmapper.retrieval.cache import EmbeddingCache
from skillmapper.retrieval.embeddings import Embedder
from skillmapper.retrieval.vector_store import METADATA_COLLECTION, NAIVE_COLLECTION

# Two disjoint text sets so the parallelism benchmark's "serial" and
# "parallel" runs both hit Ollama for real (no cache hits leaking in and
# making the comparison meaningless).
_SAMPLE_TEXTS_A = [
    f"benchmark sample query variant-a-{i} about distributed systems and machine learning"
    for i in range(8)
]
_SAMPLE_TEXTS_B = [
    f"benchmark sample query variant-b-{i} about retrieval augmented generation and hybrid search"
    for i in range(8)
]


def _fresh_embedder() -> Embedder:
    # unique cache subdir per call -- guarantees a cold cache each time this
    # benchmark runs, rather than being polluted by a previous run's cache.
    cache = EmbeddingCache(
        settings.cache_dir / f"bench_{uuid.uuid4().hex[:8]}", settings.ollama_embed_model
    )
    return Embedder(cache=cache)


def benchmark_cache() -> dict:
    """Same texts, cold cache vs warm cache -- isolates the caching win."""
    embedder = _fresh_embedder()

    start = time.perf_counter()
    for text in _SAMPLE_TEXTS_A:
        embedder.embed(text, "query")
    cold_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for text in _SAMPLE_TEXTS_A:
        embedder.embed(text, "query")
    warm_seconds = time.perf_counter() - start

    return {
        "n_texts": len(_SAMPLE_TEXTS_A),
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "speedup": cold_seconds / warm_seconds if warm_seconds > 0 else float("inf"),
    }


def benchmark_parallelism(concurrency: int = 4) -> dict:
    """Disjoint text sets, both cold-cache -- isolates the async fan-out
    win on genuine Ollama calls (this is the stage the `retrieve` node
    actually pays for: N per-requirement embedding calls).
    """
    serial_embedder = _fresh_embedder()
    start = time.perf_counter()
    for text in _SAMPLE_TEXTS_A:
        serial_embedder.embed(text, "query")
    serial_seconds = time.perf_counter() - start

    parallel_embedder = _fresh_embedder()
    start = time.perf_counter()
    asyncio.run(parallel_embedder.aembed_batch(_SAMPLE_TEXTS_B, "query", concurrency=concurrency))
    parallel_seconds = time.perf_counter() - start

    return {
        "n_texts": len(_SAMPLE_TEXTS_A),
        "concurrency": concurrency,
        "serial_seconds": serial_seconds,
        "parallel_seconds": parallel_seconds,
        "speedup": serial_seconds / parallel_seconds if parallel_seconds > 0 else float("inf"),
    }


def benchmark_storage() -> dict:
    """Chunk count + serialized text/embedding bytes for each collection
    variant on this repo's demo corpus -- framed as this repo's measured
    numbers, not a claim to reproduce a differently-scaled original dataset.
    """
    client = vector_store.get_client()
    result = {}
    for name in (NAIVE_COLLECTION, METADATA_COLLECTION):
        collection = vector_store.get_collection(client, name)
        entries = vector_store.get_all_chunks(collection)
        n_chunks = len(entries)
        text_bytes = sum(len(e["text"].encode("utf-8")) for e in entries)
        embedding_bytes = n_chunks * 768 * 4  # nomic-embed-text: 768-dim float32
        result[name] = {
            "n_chunks": n_chunks,
            "text_bytes": text_bytes,
            "embedding_bytes": embedding_bytes,
            "total_bytes": text_bytes + embedding_bytes,
        }
    return result


def run_all_benchmarks() -> dict:
    return {
        "cache": benchmark_cache(),
        "parallelism": benchmark_parallelism(),
        "storage": benchmark_storage(),
    }
