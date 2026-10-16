from __future__ import annotations

import json
import re

from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from skillmapper.config import settings

Verdict = str  # kept loose here; skillmapper.models.Verdict enforces the Literal at the boundary


class ComparisonDraft(BaseModel):
    verdict: Verdict
    overall_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    reasoning: str
    self_reported_confidence: float


def get_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


def generate_comparison(
    llm: ChatOllama, prompt: str, max_parse_retries: int = 2
) -> ComparisonDraft:
    """Ask the LLM for a structured ComparisonDraft.

    Tries LangChain's structured-output path first (json_schema mode). An 8B
    local model occasionally emits malformed/truncated JSON even in that
    mode, so on any failure we fall back to a manual "respond with only
    JSON" prompt + hand parse, retried up to `max_parse_retries` times. This
    fallback is a required component, not a contingency.
    """
    structured_llm = llm.with_structured_output(ComparisonDraft, method="json_schema")
    try:
        result = structured_llm.invoke(prompt)
        return result if isinstance(result, ComparisonDraft) else ComparisonDraft(**result)
    except Exception:  # noqa: BLE001, S110 -- deliberately broad, see docstring
        pass

    last_error: Exception | None = None
    strict_prompt = prompt + "\n\nRespond with ONLY a single JSON object. No prose, no markdown fences."
    for _ in range(max_parse_retries + 1):
        try:
            raw = llm.invoke(strict_prompt)
            data = _extract_json(raw.content)
            return ComparisonDraft(**data)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_error = e

    raise RuntimeError(
        f"failed to get a valid comparison after {max_parse_retries + 1} manual-parse attempts"
    ) from last_error
