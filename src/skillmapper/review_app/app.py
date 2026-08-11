from __future__ import annotations

import streamlit as st

from skillmapper.agent.runtime import AgentRuntime
from skillmapper.config import settings
from skillmapper.ingestion.loaders import load_all_job_descriptions, load_all_resumes
from skillmapper.models import EvaluationResult, ReviewerFeedback
from skillmapper.storage import repository

st.set_page_config(page_title="SkillMapper Reviewer", layout="wide")


@st.cache_resource
def _load_corpus():
    resumes = {r.id: r for r in load_all_resumes(settings.synthetic_resumes_dir)}
    jobs = {j.id: j for j in load_all_job_descriptions(settings.synthetic_jobs_dir)}
    return resumes, jobs


@st.cache_resource
def _get_runtime() -> AgentRuntime:
    # Loads Chroma + BM25 + the Ollama LLM client -- built once per Streamlit
    # process and reused, since it's not cheap.
    return AgentRuntime()


def _evidence_by_section(result: EvaluationResult) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for item in result.evidence:
        grouped.setdefault(item.section, []).append(item)
    return grouped


def _render_evaluation_card(result: EvaluationResult, resumes: dict, jobs: dict, key_prefix: str):
    resume = resumes.get(result.resume_id)
    job = jobs.get(result.job_id)
    candidate = resume.candidate_name if resume else result.resume_id
    job_title = f"{job.title} at {job.company}" if job else result.job_id

    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 2])
        col1.markdown(f"**{candidate}** -> {job_title}")
        col2.markdown(f"Verdict: `{result.verdict}` ({result.overall_score:.2f})")
        col3.markdown(f"Confidence: `{result.confidence:.2f}` (retries: {result.retries_used})")

        if result.status == "escalated":
            st.warning(f"Escalated: {result.escalation_reason}")

        st.markdown(f"**Reasoning:** {result.reasoning}")

        mcol, xcol = st.columns(2)
        mcol.markdown("**Matched:** " + (", ".join(result.matched_skills) or "-"))
        xcol.markdown("**Missing:** " + (", ".join(result.missing_skills) or "-"))

        with st.expander(f"Evidence ({len(result.evidence)} items, by section)"):
            for section, items in _evidence_by_section(result).items():
                st.markdown(f"_{section}_")
                for item in items:
                    st.markdown(f"- ({item.score:.3f}) {item.text}")

        existing_feedback = repository.list_feedback_for_evaluation(result.id)
        for fb in existing_feedback:
            st.info(f"Reviewed by {fb.reviewer_name}: **{fb.decision}**" + (f" -- {fb.comments}" if fb.comments else ""))

        if result.status == "escalated" and not existing_feedback:
            with st.form(key=f"{key_prefix}-form-{result.id}"):
                reviewer_name = st.text_input("Reviewer name", key=f"{key_prefix}-name-{result.id}")
                decision = st.radio(
                    "Decision", ["approve", "override", "reject"], horizontal=True,
                    key=f"{key_prefix}-decision-{result.id}",
                )
                corrected_verdict = None
                if decision == "override":
                    corrected_verdict = st.selectbox(
                        "Corrected verdict", ["strong_match", "partial_match", "weak_match"],
                        key=f"{key_prefix}-verdict-{result.id}",
                    )
                comments = st.text_area("Comments", key=f"{key_prefix}-comments-{result.id}")
                submitted = st.form_submit_button("Submit feedback")

                if submitted:
                    if not reviewer_name:
                        st.error("Reviewer name is required.")
                    else:
                        repository.save_feedback(
                            ReviewerFeedback(
                                evaluation_id=result.id,
                                reviewer_name=reviewer_name,
                                decision=decision,
                                corrected_verdict=corrected_verdict,
                                comments=comments or None,
                            )
                        )
                        st.success("Feedback saved.")
                        st.rerun()

            if st.button("Re-run comparison", key=f"{key_prefix}-rerun-{result.id}"):
                with st.spinner("Re-running agent (this can take a while on a local model)..."):
                    _get_runtime().compare_resume_to_job(
                        result.resume_id, result.job_id, force_rerun=True
                    )
                st.rerun()


def main() -> None:
    resumes, jobs = _load_corpus()

    st.title("SkillMapper Reviewer")

    with st.sidebar:
        st.header("Filters")
        resume_filter = st.selectbox("Resume", ["(any)"] + sorted(resumes), key="resume_filter")
        job_filter = st.selectbox("Job", ["(any)"] + sorted(jobs), key="job_filter")

    resume_id = None if resume_filter == "(any)" else resume_filter
    job_id = None if job_filter == "(any)" else job_filter

    queue_tab, history_tab = st.tabs(["Escalation Queue", "History"])

    with queue_tab:
        escalated = repository.query_evaluations(
            resume_id=resume_id, job_id=job_id, status="escalated"
        )
        if not escalated:
            st.info("No escalated evaluations awaiting review.")
        for result in escalated:
            _render_evaluation_card(result, resumes, jobs, key_prefix="queue")

    with history_tab:
        all_results = repository.query_evaluations(resume_id=resume_id, job_id=job_id)
        if not all_results:
            st.info("No evaluations yet -- run the agent via scripts/run_agent.py or the MCP tool first.")
        for result in all_results:
            _render_evaluation_card(result, resumes, jobs, key_prefix="history")


if __name__ == "__main__":
    main()
