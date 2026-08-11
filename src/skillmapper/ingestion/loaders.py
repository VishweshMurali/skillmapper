from __future__ import annotations

import re
from pathlib import Path

from skillmapper.models import JobDescription, Resume

_TITLE_RE = re.compile(r"^#\s+(.*)$", re.MULTILINE)


def _first_heading(raw_text: str) -> str:
    match = _TITLE_RE.search(raw_text)
    return match.group(1).strip() if match else ""


def load_resume(path: Path) -> Resume:
    raw_text = path.read_text(encoding="utf-8")
    candidate_name = _first_heading(raw_text) or path.stem
    return Resume(
        id=path.stem,
        candidate_name=candidate_name,
        source_path=str(path),
        raw_text=raw_text,
    )


def load_job_description(path: Path) -> JobDescription:
    raw_text = path.read_text(encoding="utf-8")
    heading = _first_heading(raw_text) or path.stem
    if "—" in heading:  # em dash: "Title — Company"
        title, company = (part.strip() for part in heading.split("—", 1))
    else:
        title, company = heading, ""
    return JobDescription(
        id=path.stem,
        title=title,
        company=company,
        source_path=str(path),
        raw_text=raw_text,
    )


def load_all_resumes(resumes_dir: Path) -> list[Resume]:
    return [load_resume(p) for p in sorted(resumes_dir.glob("*.md"))]


def load_all_job_descriptions(jobs_dir: Path) -> list[JobDescription]:
    return [load_job_description(p) for p in sorted(jobs_dir.glob("*.md"))]
