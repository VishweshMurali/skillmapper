"""This test literally documents the bug the portfolio's debugging story
fixed: naive whole-document chunking lets a single chunk straddle two
resume sections, separating a skill or achievement from the section that
gives it context. See test_chunking.py for the metadata-aware chunker,
which cannot exhibit this behavior by construction.
"""

from skillmapper.ingestion.naive_chunking import (
    NAIVE_SECTION_LABEL,
    chunk_document_naive,
)

BOUNDARY_FIXTURE = """# Jane Doe

## Experience

Worked at AcmeCorpXYZ building distributed systems for five years.

## Skills

Knows PythonSkillMarker and SQL very well.
"""


def test_naive_chunker_straddles_section_boundary():
    chunks = chunk_document_naive("doc1", BOUNDARY_FIXTURE, "resume", chunk_size=40, chunk_overlap=0)
    straddling = [c for c in chunks if "AcmeCorpXYZ" in c.text and "PythonSkillMarker" in c.text]
    assert straddling, "expected naive chunking to merge Experience and Skills into one chunk"


def test_naive_chunker_has_no_section_metadata():
    chunks = chunk_document_naive("doc1", BOUNDARY_FIXTURE, "resume")
    assert all(c.section == NAIVE_SECTION_LABEL for c in chunks)
