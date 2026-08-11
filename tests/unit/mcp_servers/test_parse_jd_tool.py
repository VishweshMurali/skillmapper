from parse_jd_tool import parse_jd

JOB_TEXT = """# Senior Backend Engineer — Acme Corp

## Overview

We build payments infrastructure.

## Requirements

5+ years of Python. Experience with Kubernetes.

## Qualifications

AWS certification preferred.
"""

JOB_NO_REQUIREMENTS_TEXT = """# Data Analyst — Acme Corp

## Overview

We analyze things.
"""


def test_parse_jd_returns_title_company_and_sections(tmp_path):
    (tmp_path / "job-01.md").write_text(JOB_TEXT, encoding="utf-8")

    result = parse_jd(tmp_path, "job-01")

    assert result.job_id == "job-01"
    assert result.title == "Senior Backend Engineer"
    assert result.company == "Acme Corp"
    assert [b.section for b in result.sections] == ["overview", "requirements", "qualifications"]


def test_parse_jd_requirement_blocks_only_include_requirements_and_qualifications(tmp_path):
    (tmp_path / "job-01.md").write_text(JOB_TEXT, encoding="utf-8")

    result = parse_jd(tmp_path, "job-01")

    assert {b.section for b in result.requirement_blocks} == {"requirements", "qualifications"}
    assert any("Kubernetes" in b.text for b in result.requirement_blocks)


def test_parse_jd_falls_back_to_all_sections_when_no_requirements_section(tmp_path):
    (tmp_path / "job-02.md").write_text(JOB_NO_REQUIREMENTS_TEXT, encoding="utf-8")

    result = parse_jd(tmp_path, "job-02")

    assert result.requirement_blocks == result.sections
    assert [b.section for b in result.requirement_blocks] == ["overview"]
