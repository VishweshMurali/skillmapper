from __future__ import annotations

from llama_index.core.node_parser import SentenceSplitter

from skillmapper.ingestion.section_splitter import split_into_sections
from skillmapper.models import Chunk, DocType

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 20


def chunk_document(
    doc_id: str,
    raw_text: str,
    doc_type: DocType,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Metadata-aware chunking: split into section blocks first (Experience,
    Skills, Education, ...), then run the sentence splitter *independently
    within each section*. A chunk can never contain text from two different
    sections because the splitter never sees more than one section's text
    at a time.
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    blocks = split_into_sections(raw_text, doc_type)

    chunks: list[Chunk] = []
    for block in blocks:
        pieces = splitter.split_text(block.text)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=Chunk.make_id(doc_id, block.section, i),
                    doc_id=doc_id,
                    doc_type=doc_type,
                    section=block.section,
                    chunk_index=i,
                    text=piece,
                )
            )
    return chunks
