import compare_resume_jd_tool
from compare_resume_jd_tool import compare_resume_jd

from skillmapper.agent.llm import ComparisonDraft
from skillmapper.models import EvidenceItem, JobDescription, Resume

RESUME = Resume(id="r1", candidate_name="Test Candidate", source_path="x", raw_text="resume text")
JOB = JobDescription(id="j1", title="Engineer", company="Acme", source_path="y", raw_text="job text")


def _stub_generate_comparison(monkeypatch, **overrides):
    defaults = {
        "verdict": "partial_match", "overall_score": 0.5, "matched_skills": ["Python"],
        "missing_skills": [], "reasoning": "stub reasoning", "self_reported_confidence": 0.5,
    }
    defaults.update(overrides)

    captured = {}

    def fake(llm, prompt, max_parse_retries=2):
        captured["llm"] = llm
        captured["prompt"] = prompt
        return ComparisonDraft(**defaults)

    monkeypatch.setattr(compare_resume_jd_tool, "generate_comparison", fake)
    return captured


def test_compare_resume_jd_returns_draft_fields_plus_ids(monkeypatch):
    _stub_generate_comparison(monkeypatch, verdict="strong_match", overall_score=0.9)

    result = compare_resume_jd(
        llm=None, resumes_by_id={"r1": RESUME}, jobs_by_id={"j1": JOB},
        resume_id="r1", job_id="j1", evidence=[],
    )

    assert result.resume_id == "r1"
    assert result.job_id == "j1"
    assert result.verdict == "strong_match"
    assert result.overall_score == 0.9


def test_compare_resume_jd_prompt_includes_job_and_evidence(monkeypatch):
    captured = _stub_generate_comparison(monkeypatch)
    evidence = [
        EvidenceItem(
            chunk_id="r1::skills::0", doc_id="r1", section="skills",
            text="Expert in Python.", score=0.03, supports_requirement="Python",
        )
    ]

    compare_resume_jd(
        llm=None, resumes_by_id={"r1": RESUME}, jobs_by_id={"j1": JOB},
        resume_id="r1", job_id="j1", evidence=evidence,
    )

    prompt = captured["prompt"]
    assert "Engineer at Acme" in prompt
    assert "Test Candidate" in prompt
    assert "[skills] Expert in Python." in prompt


def test_compare_resume_jd_handles_no_evidence(monkeypatch):
    captured = _stub_generate_comparison(monkeypatch)

    compare_resume_jd(
        llm=None, resumes_by_id={"r1": RESUME}, jobs_by_id={"j1": JOB},
        resume_id="r1", job_id="j1", evidence=[],
    )

    assert "(no evidence retrieved)" in captured["prompt"]
