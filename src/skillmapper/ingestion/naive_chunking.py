from __future__ import annotations

from llama_index.core.node_parser import SentenceSplitter

from skillmapper.ingestion.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from skillmapper.models import Chunk, DocType

NAIVE_SECTION_LABEL = "unsectioned"


def chunk_document_naive(
    doc_id: str,
    raw_text: str,
    doc_type: DocType,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Naive chunking: the same sentence splitter, run once over the whole
    raw document with no section pre-split. Chunking-boundary-awareness is
    the only variable this differs from `chunking.chunk_document` by — used
    to reproduce the RAGAS before/after comparison in `eval/chunking_comparison.py`.
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces = splitter.split_text(raw_text)

    return [
        Chunk(
            id=Chunk.make_id(doc_id, NAIVE_SECTION_LABEL, i),
            doc_id=doc_id,
            doc_type=doc_type,
            section=NAIVE_SECTION_LABEL,
            chunk_index=i,
            text=piece,
        )
        for i, piece in enumerate(pieces)
    ]
