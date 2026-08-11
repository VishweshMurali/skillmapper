from __future__ import annotations

from pathlib import Path

from skillmapper.ingestion.loaders import load_job_description
from skillmapper.ingestion.section_splitter import split_into_sections
from skillmapper.orchestration.schemas import ParseJdResult, RequirementBlock

# Same section names agent/nodes.py:_requirement_texts treats as "requirements" --
# a job's requirement/qualification section blocks, not individual bullets (nothing
# in this repo splits a requirements section further than that).
REQUIREMENT_SECTIONS = {"requirements", "qualifications"}


def parse_jd(jobs_dir: Path, job_id: str) -> ParseJdResult:
    job = load_job_description(jobs_dir / f"{job_id}.md")
    blocks = [
        RequirementBlock(section=b.section, heading=b.heading, text=b.text)
        for b in split_into_sections(job.raw_text, "job_description")
    ]
    requirement_blocks = [b for b in blocks if b.section in REQUIREMENT_SECTIONS]
    return ParseJdResult(
        job_id=job.id,
        title=job.title,
        company=job.company,
        sections=blocks,
        requirement_blocks=requirement_blocks or blocks,
    )
