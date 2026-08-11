from __future__ import annotations

import re
from dataclasses import dataclass

from skillmapper.models import DocType, JobSection, ResumeSection

_HEADING_RE = re.compile(r"^#{1,3}\s+(.*?)\s*$", re.MULTILINE)

_RESUME_ALIASES: dict[ResumeSection, tuple[str, ...]] = {
    ResumeSection.SUMMARY: ("summary", "profile", "objective", "about"),
    ResumeSection.EXPERIENCE: ("experience", "employment", "work history"),
    ResumeSection.SKILLS: ("skills", "technical skills", "technologies"),
    ResumeSection.EDUCATION: ("education", "academic"),
    ResumeSection.PROJECTS: ("projects", "portfolio"),
    ResumeSection.CERTIFICATIONS: ("certification", "licenses"),
}

_JOB_ALIASES: dict[JobSection, tuple[str, ...]] = {
    JobSection.OVERVIEW: ("overview", "about", "role summary", "the role"),
    JobSection.RESPONSIBILITIES: ("responsibilit", "what you'll do", "duties"),
    JobSection.REQUIREMENTS: ("requirement", "must have", "minimum qualifications"),
    JobSection.QUALIFICATIONS: ("qualification", "nice to have", "preferred"),
}


def _normalize(heading: str, doc_type: DocType) -> str:
    lowered = heading.strip().lower()
    aliases = _RESUME_ALIASES if doc_type == "resume" else _JOB_ALIASES
    fallback = ResumeSection.OTHER if doc_type == "resume" else JobSection.OTHER
    for section, patterns in aliases.items():
        if any(p in lowered for p in patterns):
            return section.value
    return fallback.value


@dataclass(frozen=True)
class SectionBlock:
    section: str
    heading: str
    text: str


def split_into_sections(raw_text: str, doc_type: DocType) -> list[SectionBlock]:
    """Split a markdown document on `#`/`##`/`###` headings into section blocks.

    Text preceding the first heading (if any) is dropped if blank, otherwise
    kept as an OTHER block so no content is silently lost.
    """
    matches = list(_HEADING_RE.finditer(raw_text))
    if not matches:
        fallback = ResumeSection.OTHER if doc_type == "resume" else JobSection.OTHER
        return [SectionBlock(section=fallback.value, heading="", text=raw_text.strip())]

    blocks: list[SectionBlock] = []

    preamble = raw_text[: matches[0].start()].strip()
    if preamble:
        fallback = ResumeSection.OTHER if doc_type == "resume" else JobSection.OTHER
        blocks.append(SectionBlock(section=fallback.value, heading="", text=preamble))

    for i, match in enumerate(matches):
        heading = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        if not body:
            continue
        blocks.append(
            SectionBlock(section=_normalize(heading, doc_type), heading=heading, text=body)
        )

    return blocks
