from skillmapper.ingestion.section_splitter import split_into_sections
from skillmapper.models import JobSection, ResumeSection

RESUME_TEXT = """# Jane Doe

## Summary

A short summary.

## Experience

Did things at a company. Did more things.

## Skills

Python, SQL.
"""

JOB_TEXT = """# Some Title — Some Company

## Overview

We do things.

## Requirements

You need Python.
"""


def test_splits_resume_into_expected_sections():
    blocks = split_into_sections(RESUME_TEXT, "resume")
    sections = [b.section for b in blocks]
    assert sections == [
        ResumeSection.SUMMARY.value,
        ResumeSection.EXPERIENCE.value,
        ResumeSection.SKILLS.value,
    ]


def test_section_text_does_not_include_heading_or_neighbors():
    blocks = split_into_sections(RESUME_TEXT, "resume")
    experience = next(b for b in blocks if b.section == ResumeSection.EXPERIENCE.value)
    assert "Did things at a company" in experience.text
    assert "Summary" not in experience.text
    assert "Python, SQL" not in experience.text


def test_splits_job_description_into_expected_sections():
    blocks = split_into_sections(JOB_TEXT, "job_description")
    sections = [b.section for b in blocks]
    assert sections == [JobSection.OVERVIEW.value, JobSection.REQUIREMENTS.value]


def test_unrecognized_heading_falls_back_to_other():
    text = "# Name\n\n## Hobbies\n\nI like hiking.\n"
    blocks = split_into_sections(text, "resume")
    assert blocks[0].section == ResumeSection.OTHER.value


def test_document_with_no_headings_returns_single_other_block():
    text = "Just some plain text with no markdown headings at all."
    blocks = split_into_sections(text, "resume")
    assert len(blocks) == 1
    assert blocks[0].section == ResumeSection.OTHER.value
    assert blocks[0].text == text
