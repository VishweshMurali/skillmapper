from __future__ import annotations

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from skillmapper.agent import nodes
from skillmapper.agent.state import AgentState
from skillmapper.config import settings
from skillmapper.models import JobDescription, Resume
from skillmapper.retrieval.hybrid import HybridRetriever

RECURSION_LIMIT = 15


def route_after_confidence_check(threshold: float):
    def route(state: AgentState) -> str:
        if state["confidence"] >= threshold:
            return "finalize"
        if state["retry_count"] < state["max_retries"]:
            return "retry"
        return "escalate"

    return route


def build_graph(
    retriever: HybridRetriever,
    llm: ChatOllama,
    resumes_by_id: dict[str, Resume],
    jobs_by_id: dict[str, JobDescription],
    confidence_threshold: float | None = None,
):
    threshold = (
        confidence_threshold if confidence_threshold is not None
        else settings.agent_confidence_threshold
    )

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", nodes.make_retrieve_node(retriever))
    graph.add_node("compare", nodes.make_compare_node(llm, resumes_by_id, jobs_by_id))
    graph.add_node("confidence_check", nodes.confidence_check)
    graph.add_node("increment_retry", nodes.increment_retry)
    graph.add_node("finalize", nodes.make_finalize_node())
    graph.add_node("escalate", nodes.make_escalate_node())

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "compare")
    graph.add_edge("compare", "confidence_check")
    graph.add_conditional_edges(
        "confidence_check",
        route_after_confidence_check(threshold),
        {"finalize": "finalize", "retry": "increment_retry", "escalate": "escalate"},
    )
    graph.add_edge("increment_retry", "retrieve")
    graph.add_edge("finalize", END)
    graph.add_edge("escalate", END)

    return graph.compile()


def invoke_graph(compiled_graph, initial_state: AgentState) -> AgentState:
    return compiled_graph.invoke(initial_state, config={"recursion_limit": RECURSION_LIMIT})
