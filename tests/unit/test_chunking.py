from skillmapper.ingestion.chunking import chunk_document
from skillmapper.models import ResumeSection

# Small enough total token count that a chunk_size=40 splitter run over the
# *whole* document (no section pre-split) merges both sections into one
# chunk -- see test_naive_chunking.py for the naive counterpart. Here the
# metadata-aware chunker must never let that happen, by construction.
BOUNDARY_FIXTURE = """# Jane Doe

## Experience

Worked at AcmeCorpXYZ building distributed systems for five years.

## Skills

Knows PythonSkillMarker and SQL very well.
"""


def test_chunks_never_straddle_a_section_boundary():
    chunks = chunk_document("doc1", BOUNDARY_FIXTURE, "resume", chunk_size=40, chunk_overlap=0)
    for chunk in chunks:
        assert not ("AcmeCorpXYZ" in chunk.text and "PythonSkillMarker" in chunk.text), (
            f"chunk spans both Experience and Skills content: {chunk.text!r}"
        )


def test_each_chunk_tagged_with_its_source_section():
    chunks = chunk_document("doc1", BOUNDARY_FIXTURE, "resume", chunk_size=40, chunk_overlap=0)
    sections = {c.section for c in chunks}
    assert sections == {ResumeSection.EXPERIENCE.value, ResumeSection.SKILLS.value}
    for chunk in chunks:
        if chunk.section == ResumeSection.EXPERIENCE.value:
            assert "AcmeCorpXYZ" in chunk.text
        if chunk.section == ResumeSection.SKILLS.value:
            assert "PythonSkillMarker" in chunk.text


def test_chunk_ids_are_deterministic_and_idempotent():
    chunks_a = chunk_document("doc1", BOUNDARY_FIXTURE, "resume")
    chunks_b = chunk_document("doc1", BOUNDARY_FIXTURE, "resume")
    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]
    assert len(chunks_a) == len({c.id for c in chunks_a})
