from __future__ import annotations

from langchain_ollama import ChatOllama

# Byte-identical prompt template to agent/nodes.py:make_compare_node -- this
# server replaces that node's role in the pipeline, not its behavior.
from skillmapper.agent.llm import generate_comparison
from skillmapper.models import EvidenceItem, JobDescription, Resume
from skillmapper.orchestration.schemas import CompareResult


def compare_resume_jd(
    llm: ChatOllama,
    resumes_by_id: dict[str, Resume],
    jobs_by_id: dict[str, JobDescription],
    resume_id: str,
    job_id: str,
    evidence: list[EvidenceItem],
) -> CompareResult:
    job = jobs_by_id[job_id]
    resume = resumes_by_id[resume_id]

    evidence_text = "\n".join(f"- [{e.section}] {e.text}" for e in evidence) or "(no evidence retrieved)"

    prompt = f"""You are evaluating whether a candidate's resume matches a job description.

Job Title: {job.title} at {job.company}
Job Description:
{job.raw_text}

Candidate: {resume.candidate_name}
Retrieved resume evidence (via hybrid search against the job's requirements):
{evidence_text}

Based ONLY on the evidence above, evaluate the match and respond with a JSON object with fields:
- verdict: one of "strong_match", "partial_match", "weak_match"
- overall_score: float between 0 and 1
- matched_skills: list of requirements/skills clearly supported by the evidence
- missing_skills: list of requirements not supported by the evidence
- reasoning: 2-4 sentence explanation citing the evidence
- self_reported_confidence: float between 0 and 1, your own confidence in this assessment
"""
    draft = generate_comparison(llm, prompt)
    return CompareResult(resume_id=resume_id, job_id=job_id, **draft.model_dump())
