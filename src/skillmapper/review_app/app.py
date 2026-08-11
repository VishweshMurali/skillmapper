from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from skillmapper.agent.runtime import AgentRuntime
from skillmapper.config import settings
from skillmapper.ingestion.chunking import chunk_document
from skillmapper.ingestion.loaders import load_all_job_descriptions, load_all_resumes
from skillmapper.ingestion.naive_chunking import chunk_document_naive
from skillmapper.ingestion.pipeline import bm25_path_for, delete_document, ingest_sync
from skillmapper.models import EvaluationResult, ReviewerFeedback
from skillmapper.retrieval.bm25_index import BM25Index
from skillmapper.retrieval.vector_store import (
    METADATA_COLLECTION,
    NAIVE_COLLECTION,
    get_client,
    get_collection,
)
from skillmapper.storage import repository

st.set_page_config(page_title="SkillMapper Reviewer", layout="wide")

_DOC_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


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


def _reingest_and_refresh(doc_type: str, doc_id: str, raw_text: str, source_path: Path) -> None:
    """Write `raw_text` to `source_path`, then delete+re-ingest `doc_id` into
    both collections and refresh the runtime's in-memory BM25 index and
    resume/job corpus in place, so the change is visible without a restart.
    """
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(raw_text, encoding="utf-8")

    for collection_name in (METADATA_COLLECTION, NAIVE_COLLECTION):
        delete_document(collection_name, doc_id)
    ingest_sync(METADATA_COLLECTION, chunk_document)
    ingest_sync(NAIVE_COLLECTION, chunk_document_naive)

    runtime = _get_runtime()
    runtime.retriever.bm25_index = BM25Index.load(bm25_path_for(METADATA_COLLECTION))
    runtime.resumes_by_id.clear()
    runtime.resumes_by_id.update({r.id: r for r in load_all_resumes(settings.synthetic_resumes_dir)})
    runtime.jobs_by_id.clear()
    runtime.jobs_by_id.update(
        {j.id: j for j in load_all_job_descriptions(settings.synthetic_jobs_dir)}
    )

    _load_corpus.clear()


def _chunk_counts_for_doc(collection, doc_id: str) -> dict[str, int]:
    raw = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in raw["metadatas"]:
        counts[meta["section"]] = counts.get(meta["section"], 0) + 1
    return counts


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


def _render_documents_tab(resumes: dict, jobs: dict):
    st.subheader("Ingested documents")
    st.caption(
        "Every resume and job description currently chunked into the "
        f"`{METADATA_COLLECTION}` collection, with its section breakdown."
    )

    client = get_client()
    meta_collection = get_collection(client, METADATA_COLLECTION)

    rows = []
    for r in resumes.values():
        counts = _chunk_counts_for_doc(meta_collection, r.id)
        rows.append(
            {
                "id": r.id,
                "type": "resume",
                "title": r.candidate_name,
                "chunks": sum(counts.values()),
                "sections": ", ".join(sorted(counts)) or "-",
            }
        )
    for j in jobs.values():
        counts = _chunk_counts_for_doc(meta_collection, j.id)
        title = f"{j.title} at {j.company}" if j.company else j.title
        rows.append(
            {
                "id": j.id,
                "type": "job_description",
                "title": title,
                "chunks": sum(counts.values()),
                "sections": ", ".join(sorted(counts)) or "-",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Edit an existing document")

    doc_options = ["(select a document)"] + [f"{row['type']}: {row['id']}" for row in rows]
    selected = st.selectbox("Document", doc_options, key="edit_doc_select")

    if selected != "(select a document)":
        sel_type, sel_id = selected.split(": ", 1)
        doc = resumes[sel_id] if sel_type == "resume" else jobs[sel_id]

        edited_text = st.text_area(
            "Markdown content", value=doc.raw_text, height=350, key=f"edit_text_{sel_type}_{sel_id}"
        )
        if st.button("Save changes & re-ingest", key=f"edit_save_{sel_type}_{sel_id}"):
            with st.spinner("Re-ingesting (chunking + embedding + rebuilding indexes)..."):
                _reingest_and_refresh(sel_type, sel_id, edited_text, Path(doc.source_path))
            st.success(f"Saved and re-ingested '{sel_id}'.")
            st.rerun()

    st.divider()
    st.subheader("Add a new document")

    new_doc_type = st.radio(
        "Document type", ["resume", "job_description"], horizontal=True, key="new_doc_type"
    )
    uploaded = st.file_uploader(
        "Upload a markdown file (optional -- or paste content below)",
        type=["md", "txt"],
        key="new_doc_upload",
    )
    uploaded_text = uploaded.getvalue().decode("utf-8") if uploaded is not None else ""
    default_id = Path(uploaded.name).stem if uploaded is not None else ""

    new_doc_id = st.text_input(
        "Document id (used as the filename; letters, digits, hyphens, underscores only)",
        value=default_id,
        key="new_doc_id",
    )
    new_doc_text = st.text_area(
        "Markdown content (must start with a `# Heading` -- candidate name for resumes, "
        "\"Title — Company\" for job descriptions)",
        value=uploaded_text,
        height=250,
        key="new_doc_text",
    )

    if st.button("Ingest document", key="new_doc_submit", type="primary"):
        if not new_doc_id or not _DOC_ID_RE.match(new_doc_id):
            st.error("Document id is required and may only contain letters, digits, - and _.")
        elif not new_doc_text.strip():
            st.error("Document content is empty.")
        else:
            target_dir = (
                settings.synthetic_resumes_dir
                if new_doc_type == "resume"
                else settings.synthetic_jobs_dir
            )
            existing_ids = resumes if new_doc_type == "resume" else jobs
            if new_doc_id in existing_ids:
                st.warning(f"'{new_doc_id}' already exists as a {new_doc_type} -- overwriting it.")
            with st.spinner("Ingesting (chunking + embedding + rebuilding indexes)..."):
                _reingest_and_refresh(
                    new_doc_type, new_doc_id, new_doc_text, target_dir / f"{new_doc_id}.md"
                )
            st.success(f"Ingested '{new_doc_id}' as a {new_doc_type}.")
            st.rerun()


def _render_retrieval_tab():
    st.subheader("Test hybrid retrieval")
    st.caption(
        "Runs the same BM25 + dense (Chroma) retriever, fused with Reciprocal Rank Fusion, "
        "that the agent's `retrieve` node and the `search_documents`/`retrieve_evidence` MCP "
        "tools use -- against the live metadata-aware collection."
    )

    query = st.text_input(
        "Query", key="retrieval_query", placeholder="e.g. Kubernetes production experience"
    )
    col1, col2, col3, col4 = st.columns(4)
    doc_type = col1.selectbox("Doc type", ["any", "resume", "job_description"], key="retrieval_doc_type")
    doc_id = col2.text_input("Filter by doc id (optional)", key="retrieval_doc_id")
    section = col3.text_input("Filter by section (optional)", key="retrieval_section")
    top_k = col4.number_input("top_k", min_value=1, max_value=25, value=5, key="retrieval_top_k")

    if st.button("Search", key="retrieval_search_btn", type="primary"):
        if not query.strip():
            st.error("Enter a query.")
        else:
            runtime = _get_runtime()
            with st.spinner("Searching..."):
                hits = runtime.retriever.search(
                    query,
                    k=int(top_k),
                    doc_type=None if doc_type == "any" else doc_type,
                    doc_id=doc_id or None,
                    section=section or None,
                )
            st.session_state["retrieval_last_hits"] = hits

    hits = st.session_state.get("retrieval_last_hits")
    if hits is not None:
        if not hits:
            st.info("No results.")
        for hit in hits:
            meta = hit["metadata"]
            with st.container(border=True):
                st.markdown(
                    f"**{meta['doc_id']}** ({meta['doc_type']}) · `{meta['section']}` "
                    f"· RRF score {hit['score']:.4f}"
                )
                st.write(hit["text"])


def _render_score_tab(resumes: dict, jobs: dict):
    st.subheader("Score a resume against a job description")
    st.caption(
        "Runs the full LangGraph agent (retrieve -> compare -> confidence-check, with "
        "retry/escalation) and persists the result -- it will also show up in History below."
    )

    if not resumes or not jobs:
        st.info("No resumes/job descriptions ingested yet -- add some in the Documents tab.")
        return

    col1, col2, col3 = st.columns([2, 2, 1])
    resume_id = col1.selectbox(
        "Resume", sorted(resumes),
        format_func=lambda rid: f"{resumes[rid].candidate_name} ({rid})",
        key="score_resume",
    )
    job_id = col2.selectbox(
        "Job description", sorted(jobs),
        format_func=lambda jid: f"{jobs[jid].title} ({jid})",
        key="score_job",
    )
    force_rerun = col3.checkbox(
        "Force re-run", key="score_force_rerun",
        help="Ignore any cached completed evaluation and run the agent again.",
    )

    if st.button("Run scoring agent", key="score_run_btn", type="primary"):
        with st.spinner(
            "Running retrieve -> compare -> confidence-check (can take 30s-a few minutes "
            "on a local model)..."
        ):
            result = _get_runtime().compare_resume_to_job(
                resume_id, job_id, force_rerun=force_rerun
            )
        st.session_state["score_last_result"] = result

    last = st.session_state.get("score_last_result")
    if last is not None and last.resume_id == resume_id and last.job_id == job_id:
        _render_evaluation_card(last, resumes, jobs, key_prefix="score")


def main() -> None:
    resumes, jobs = _load_corpus()

    st.title("SkillMapper Reviewer")

    with st.sidebar:
        st.header("Filters")
        resume_filter = st.selectbox("Resume", ["(any)"] + sorted(resumes), key="resume_filter")
        job_filter = st.selectbox("Job", ["(any)"] + sorted(jobs), key="job_filter")

    resume_id = None if resume_filter == "(any)" else resume_filter
    job_id = None if job_filter == "(any)" else job_filter

    documents_tab, retrieval_tab, score_tab, queue_tab, history_tab = st.tabs(
        ["Documents", "Retrieval", "Score", "Escalation Queue", "History"]
    )

    with documents_tab:
        _render_documents_tab(resumes, jobs)

    with retrieval_tab:
        _render_retrieval_tab()

    with score_tab:
        _render_score_tab(resumes, jobs)

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
            st.info("No evaluations yet -- run the agent via scripts/run_agent.py, the Score tab, or the MCP tool first.")
        for result in all_results:
            _render_evaluation_card(result, resumes, jobs, key_prefix="history")


if __name__ == "__main__":
    main()
