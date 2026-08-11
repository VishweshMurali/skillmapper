from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from skillmapper.config import settings
from skillmapper.models import Chunk

METADATA_COLLECTION = "skillmapper_chunks"
NAIVE_COLLECTION = "skillmapper_chunks_naive"


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_collection(client: chromadb.ClientAPI, name: str) -> Collection:
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(
    collection: Collection, chunks: list[Chunk], embeddings: list[list[float]]
) -> None:
    if not chunks:
        return
    collection.upsert(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_id": c.doc_id,
                "doc_type": c.doc_type,
                "section": c.section,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ],
    )


def query(
    collection: Collection,
    query_embedding: list[float],
    n_results: int,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Dense query, returns a flat list of {id, text, metadata, distance}."""
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    return [
        {"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]}
        for i in range(len(ids))
    ]


def get_all_chunks(collection: Collection) -> list[dict[str, Any]]:
    """Every chunk in the collection -- used to build the BM25 index."""
    result = collection.get(include=["documents", "metadatas"])
    ids = result["ids"]
    docs = result["documents"]
    metas = result["metadatas"]
    return [{"id": ids[i], "text": docs[i], "metadata": metas[i]} for i in range(len(ids))]
