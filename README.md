# SkillMapper

An agentic RAG system that compares resumes against job descriptions, retrieves supporting evidence via hybrid search, and produces structured evaluation results with human-in-the-loop escalation for low-confidence cases. Exposed two ways -- a single consolidated MCP server (5 tools, in-process LangGraph agent) and an alternate 5-server MCP architecture (one server per pipeline stage, driven by an orchestrator client) -- plus a shared Streamlit reviewer UI.

Runs entirely locally against [Ollama](https://ollama.com) (`llama3.1:8b` for generation, `nomic-embed-text` for embeddings) — no paid API keys required.

## Architecture

```
                         ┌─────────────────────────┐
                         │      MCP Server          │
                         │  (5 tools, stdio)         │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼────────────────────────┐
              │                       │                        │
   ┌──────────▼──────────┐  ┌─────────▼─────────┐   ┌──────────▼──────────┐
   │ search_documents      │  │ compare_resume_to_job│   │ lookup_evaluation_  │
   │ retrieve_evidence     │  │  → LangGraph agent   │   │ result / capture_   │
   │  → hybrid retriever   │  │                        │   │ reviewer_feedback   │
   └──────────┬────────────┘  └─────────┬──────────────┘   │  → SQLite repo      │
              │                          │                  └──────────┬───────────┘
              │              ┌───────────▼────────────┐               │
              │              │ retrieve → compare →     │               │
              │              │ confidence_check          │◄──────────────┘
              │              │  ├─ retry (widen k) ───┐  │
              │              │  ├─ escalate ──────────┼──┼──► Streamlit reviewer app
              │              │  └─ finalize            │  │   (approve/override/reject,
              │              └────────────┬────────────┘  │    feeds back into SQLite)
              │                            │               │
   ┌──────────▼────────────────────────────▼───────────────▼──────┐
   │      Hybrid retriever: BM25 (rank_bm25) + Chroma (dense)       │
   │      fused with Reciprocal Rank Fusion                          │
   └──────────┬─────────────────────────────────────────────────────┘
              │
   ┌──────────▼──────────────────────────────────────────┐
   │  Ingestion: LlamaIndex SentenceSplitter run per-section  │
   │  (section boundaries detected first) → embedded via     │
   │  Ollama (nomic-embed-text, content-hash cached, async)  │
   └───────────────────────────────────────────────────────┘
```

### Alternate: 5-server MCP architecture

`mcp_servers/` ports the same retry/escalate semantics to five standalone MCP servers -- one per pipeline stage -- driven by an orchestrator client (`scripts/run_orchestrator.py`) instead of in-process LangGraph nodes. It's a parallel implementation, not a replacement: both architectures share the same hybrid retriever, SQLite repo, and Streamlit app, and both are actively tested (see [Repo layout](#repo-layout)).

```
    ┌────────────────────────────────────────────────────────────────┐
    │ Orchestrator client (scripts/run_orchestrator.py)              │
    │ spawns + drives 5 servers over stdio, per session               │
    └────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ jd-parser: parse_jd(job_id)                                     │
    │ -> requirement_blocks                                           │
    └────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ for each requirement block:                                     │
    │ rag-router: decide_retrieval(retry_count)                       │
    │ -> hybrid retriever (BM25 + Chroma, RRF) -> evidence             │
    └────────────────────────────────────────────────────────────────┘
                                     │
                              (all evidence)
                                     ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ resume-comparator: compare_resume_jd                            │
    │ -> verdict, matched/missing skills, self-reported confidence    │
    └────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ scorer: score_comparison(evidence, self_reported_confidence)    │
    │ -> confidence                                                   │
    └────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    confidence >= threshold?
      ├─ yes ─────────────────────────────────────────────────► completed
      ├─ no, retries left ──► retry_count++, loop back to rag-router
      └─ no, retries exhausted ─────────────────────────────────► escalated
                                     │
                                     ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ SQLite repo (same repo as the LangGraph path above)             │
    │ + data/sessions/<session_id>.jsonl event log                    │
    │ -> feeds the same Streamlit reviewer app                        │
    └────────────────────────────────────────────────────────────────┘
                                     │
                         (optional --eval flag)
                                     ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ ragas-eval: run_ragas_eval(session_id)                          │
    │ re-judges the session's JSONL log after the fact                │
    └────────────────────────────────────────────────────────────────┘
```

Run it the same way as the LangGraph path:

```bash
uv run scripts/run_orchestrator.py resume-01-senior-swe-backend job-01-swe-backend
uv run scripts/run_orchestrator.py resume-01-senior-swe-backend job-01-swe-backend --eval  # + RAGAS-score the session
```

### Why these choices

- **Custom LangGraph `StateGraph`, not `create_react_agent`.** The agent needs an explicit retry counter (widening `k` on each attempt, not repeating the same call) and a terminal escalate-to-human path. The prebuilt ReAct agent's fixed internal routing doesn't expose hooks for either.
- **Hybrid retrieval (BM25 + dense), fused with Reciprocal Rank Fusion.** Semantic similarity alone can miss an exact certification name or tool acronym (e.g. "CKA") buried in a long skills list; BM25 catches the exact string, dense catches paraphrases. Both signals are fused rather than picking one.
- **Raw `chromadb` + `rank_bm25` instead of `llama-index-vector-stores-chroma` / `QueryFusionRetriever`.** LlamaIndex is used only for its chunking primitives. Retrieval and fusion are hand-written so the RRF math and metadata filtering are unit-testable and debuggable directly, instead of living behind a framework abstraction — the same "don't treat the pipeline as a black box" principle that drove the chunking fix below.
- **Metadata-aware, section-first chunking.** See the debugging story.

## The debugging story

The pipeline was fully orchestrated and technically functional, but wasn't producing good evaluations. Rather than guessing at the LLM prompt or the agent's routing logic, each stage was isolated and tested independently. Running RAGAS against the RAG layer alone (not the full comparison agent) showed low **faithfulness** and **context recall** — a retrieval/chunking problem, not an orchestration problem.

Inspecting the chunker: naive whole-document sentence-splitting was cutting across resume section boundaries — merging the tail of an "Experience" entry into the start of a "Skills" list, or vice versa, whenever the two were short enough to fit in one chunk window. The fix was metadata-aware chunking: split the document into sections *first* (via heading detection), then run the sentence splitter independently within each section, so a chunk can never straddle two sections by construction.

This is reproduced as a real, re-runnable comparison in this repo — not just asserted. `scripts/run_chunking_comparison.py` ingests the same corpus into two Chroma collections (naive vs. metadata-aware), runs the identical 12-question test set through both, and scores each with RAGAS under the same local judge:

Reproduce it with `uv run scripts/ingest.py` followed by `uv run scripts/run_chunking_comparison.py` (takes tens of minutes on a local 8B judge; results are cached under `data/cache/` and shown in the Streamlit app's Model Evaluation tab). Scores are not committed to the repo because they depend on your local hardware and judge model. See [Limitations](#limitations) for why an 8B local judge's absolute scores aren't comparable to GPT-4-judged benchmarks — the delta under a fixed judge is the meaningful number here.

A synthetic unit test also encodes the bug directly: `tests/unit/test_naive_chunking.py::test_naive_chunker_straddles_section_boundary` constructs a fixture where naive chunking demonstrably merges two sections into one chunk, and `tests/unit/test_chunking.py::test_chunks_never_straddle_a_section_boundary` proves the metadata-aware chunker cannot do this.

## The optimization story

Two mechanisms were implemented for real and benchmarked honestly — including where the numbers didn't match initial assumptions:

**Embedding cache** (content-hash + query/document-role keyed, via `diskcache`): a clear, dramatic win. Re-embedding the same 8 texts with a warm cache took **essentially 0s vs. 2.7s cold** (measured with `scripts/run_benchmark.py`) — a >6000x speedup, since a cache hit skips the Ollama round-trip entirely.

**Async parallel embedding** (`asyncio.gather` + `asyncio.Semaphore` fan-out, used by the agent's `retrieve` node when it queries once per JD requirement): measured **no speedup** in this environment (serial and concurrency=4 both took ~2.4s for 8 texts). This local Ollama install processes embedding requests serially by default (`OLLAMA_NUM_PARALLEL` is typically 1 unless explicitly reconfigured), so concurrent requests queue rather than overlap. The code path is real and correctly bounded (verified in `tests/unit/test_embedding_cache.py`), but on this machine the cache — not the parallelism — is what actually reduces latency. This is reported as measured, not adjusted to match a target number.

**Storage**: comparing the two Chroma collections on this repo's demo corpus (8 resumes + 3 job descriptions):

| Collection | Chunks | Text bytes | Embedding bytes | Total |
|---|---|---|---|---|
| naive | 25 | 16.4KB | 75.0KB | 91.4KB |
| metadata-aware | 54 | 14.7KB | 162.0KB | 176.7KB |

Metadata-aware chunking here uses **more** storage than naive, not less — because respecting section boundaries prevents the splitter from merging short adjacent sections into one chunk, producing more (smaller) chunks and therefore more embeddings. This is the real, honest tradeoff: the retrieval-quality gain from section-aware chunking (see the RAGAS table above) is paid for partly in embedding storage, not free. (Regenerate with `uv run scripts/run_benchmark.py`; requires `scripts/ingest.py` to have populated both collections first.)

## Quickstart

```bash
# 1. Install Ollama (https://ollama.com) and pull the two models used
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 2. Install dependencies
uv sync

# 3. Ingest the synthetic resume/JD corpus (both chunking variants)
uv run scripts/ingest.py

# 4. Run the agent once, end to end
uv run scripts/run_agent.py resume-01-senior-swe-backend job-01-swe-backend

# 5. Run the MCP server (stdio transport)
uv run scripts/run_mcp_server.py

# 6. Run the Streamlit reviewer app
uv run streamlit run src/skillmapper/review_app/app.py
```

### MCP tools

| Tool | Purpose |
|---|---|
| `search_documents` | Hybrid (BM25 + dense) search over ingested resume/JD chunks |
| `retrieve_evidence` | Resume evidence supporting one specific job requirement |
| `compare_resume_to_job` | Full agent run: retrieve → compare → retry/escalate → structured `EvaluationResult` |
| `capture_reviewer_feedback` | Record a human reviewer's decision (same path the Streamlit app uses) |
| `lookup_evaluation_result` | Query stored evaluations by id, resume, job, or status |

`compare_resume_to_job` can take **30 seconds to a few minutes** on a local 8B model (retrieval + generation, plus up to 2 retries). It short-circuits to a cached completed result on repeat calls unless `force_rerun=True`. The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) is a good way to exercise all 5 tools manually:

```bash
npx @modelcontextprotocol/inspector uv run scripts/run_mcp_server.py
```

### Streamlit app

The Streamlit app (`uv run streamlit run src/skillmapper/review_app/app.py`) is more than a reviewer queue -- it's a working front end onto the whole pipeline:

| Tab | What it does |
|---|---|
| **Documents** | Lists every ingested resume/JD with its chunk/section breakdown; upload a new one (file or pasted markdown) or edit an existing one in place. Saving deletes that document's stale chunks, re-ingests it into both Chroma collections, and reloads the in-memory BM25 index -- no restart needed. |
| **Retrieval** | Runs the same hybrid (BM25 + dense, RRF-fused) retriever the agent uses, with live query/doc-type/doc-id/section filters, so you can see exactly what evidence a given query surfaces. |
| **Score** | Pick a resume + job description and run the full LangGraph agent on demand; the result renders immediately and is persisted (it also appears in History). |
| **Escalation Queue / History** | The original reviewer workflow -- approve/override/reject escalated evaluations, browse past runs. |

### Observability (optional): LangSmith

The agent's LLM calls (`langchain_ollama.ChatOllama`) and the hybrid retriever (`@traceable`-wrapped) are LangSmith-instrumented but inert by default. Set in `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your key from smith.langchain.com>
LANGSMITH_PROJECT=skillmapper
```

and every `compare_resume_to_job` run produces a nested trace tree in LangSmith: `retrieve` (with the underlying `hybrid_retrieve` call and its RRF-fused hits) → `compare` (the exact prompt and structured-output parse) → `confidence_check` → the retry/escalate routing decision. With `LANGSMITH_TRACING` unset or false, none of this code path runs and there is no dependency on the LangSmith service.

## Testing

```bash
uv run pytest                 # fast unit suite, no Ollama required (default)
uv run pytest -m ollama       # + integration tests against a real local Ollama
```

Unit tests cover: section splitting, chunking boundary-safety (including a test that documents the naive-chunking bug directly), Reciprocal Rank Fusion math and metadata post-filtering, embedding cache hit/miss + query/document-role discrimination, LangGraph retry/escalation routing (stubbed LLM, real k-widening-per-retry assertions), SQLite repository round-trips, MCP tool functions, and RAGAS report formatting.

## Repo layout

```
src/skillmapper/
  models.py              Resume, JobDescription, Chunk, EvaluationResult, ReviewerFeedback
  config.py               pydantic-settings (Ollama host/models, thresholds, paths, optional LangSmith)
  ingestion/               section splitting, metadata-aware + naive chunkers, ingest/delete pipeline
  retrieval/               embeddings (cached), Chroma, BM25, hybrid RRF fusion (LangSmith-traceable)
  agent/                   LangGraph state/nodes/graph, LLM wrapper, runtime wiring
  storage/                 SQLite repository (evaluations, reviewer feedback)
  mcp_server/              FastMCP-style server (mcp.server.mcpserver.MCPServer) + 5 tools
  review_app/              Streamlit UI: Documents (browse/upload/edit), Retrieval (search tester),
                             Score (on-demand agent run), Escalation Queue, History
  eval/                    RAGAS harness, testset, chunking comparison, benchmark, report formatting
  orchestration/           shared schemas + JSONL session-event log used by the 5-server architecture
mcp_servers/                the alternate 5-server MCP architecture (see above), one dir per server:
  jd-parser/                 parse_jd
  rag-router/                 decide_retrieval (wraps the same hybrid retriever)
  resume-comparator/          compare_resume_jd
  scorer/                     score_comparison
  ragas-eval/                 run_ragas_eval (scores a session's JSONL log)
  orchestrator/               client that spawns + drives the 5 servers over stdio
scripts/                   CLI entry points for each of the above, incl. run_orchestrator.py
tests/unit/                fast, fully mocked (incl. tests/unit/mcp_servers/ for the 5-server tools)
tests/integration/         marked slow/ollama, require a real local Ollama (incl. the 5-server e2e path)
data/synthetic/            8 synthetic resumes + 3 job descriptions (author-generated, fictitious)
```

## Limitations

- **RAGAS judge is a local 8B model, not GPT-4.** Scores are noisier and not directly comparable to published GPT-4-judged benchmarks. The chunking-comparison story is built around the *delta* between naive and metadata-aware chunking under the same fixed judge, which is judge-independent in a way absolute scores aren't.
- **`compare_resume_to_job` is slow** (30s-minutes) on local hardware — there's no async job-polling API, just a SQLite short-circuit on repeat calls.
- **Single-user SQLite**, no auth on the Streamlit app — fine for a local portfolio demo, not for multi-tenant use.
- **Synthetic corpus is small** (8 resumes, 3 JDs) — enough to demonstrate the retrieval/chunking/agent mechanics honestly, not a claim about performance at scale.

## License

MIT -- see [LICENSE](LICENSE).
