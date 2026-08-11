from __future__ import annotations

import warnings

from langchain_ollama import OllamaEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

with warnings.catch_warnings():
    # classic ragas.metrics import path is deprecated in favor of
    # ragas.metrics.collections as of 0.4.x, but still supported and is
    # what evaluate() expects here.
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

from skillmapper.agent.llm import get_llm
from skillmapper.agent.nodes import BASE_K
from skillmapper.config import settings
from skillmapper.eval.testset import TESTSET, QAItem
from skillmapper.ingestion.pipeline import bm25_path_for
from skillmapper.retrieval import vector_store
from skillmapper.retrieval.bm25_index import BM25Index
from skillmapper.retrieval.embeddings import Embedder
from skillmapper.retrieval.hybrid import HybridRetriever

# Deliberately the same k the agent's `retrieve` node actually uses
# (skillmapper.agent.nodes.BASE_K) -- this eval exists to compare chunking
# strategies *as deployed*, not under an eval-only retrieval budget. Note
# that k means something different per chunker: naive chunking maxes out
# at 3 chunks/doc on this corpus, so k=5 always returns the full document;
# metadata-aware chunking's one-chunk-minimum-per-section design maxes out
# at 7, so k=5 can truncate it. That asymmetry is real production behavior,
# not an eval bug -- see the README's chunking-comparison section.
ANSWER_TOP_K = BASE_K

QA_PROMPT_TEMPLATE = """Answer the question using ONLY the context below. Be concise (1-3 sentences).

Context:
{context}

Question: {question}
Answer:"""


def build_retriever(collection_name: str) -> HybridRetriever:
    client = vector_store.get_client()
    collection = vector_store.get_collection(client, collection_name)
    bm25 = BM25Index.load(bm25_path_for(collection_name))
    return HybridRetriever(collection, bm25, Embedder())


def answer_question(retriever: HybridRetriever, llm, item: QAItem) -> tuple[str, list[str]]:
    hits = retriever.search(item.question, k=ANSWER_TOP_K, doc_id=item.doc_id)
    contexts = [hit["text"] for hit in hits]
    prompt = QA_PROMPT_TEMPLATE.format(
        context="\n".join(f"- {c}" for c in contexts) or "(no context retrieved)",
        question=item.question,
    )
    response = llm.invoke(prompt)
    return response.content, contexts


def build_dataset(collection_name: str, testset: list[QAItem] = TESTSET) -> EvaluationDataset:
    """Runs the actual RAG QA path (hybrid retrieval + a lightweight
    answer-generation call) for every testset question -- this evaluates
    the RAG layer itself, not the verdict-producing comparison agent.
    """
    retriever = build_retriever(collection_name)
    llm = get_llm(temperature=0)

    samples = []
    for item in testset:
        response, contexts = answer_question(retriever, llm, item)
        samples.append(
            SingleTurnSample(
                user_input=item.question,
                retrieved_contexts=contexts,
                response=response,
                reference=item.reference,
            )
        )
    return EvaluationDataset(samples=samples)


def run_ragas_eval(collection_name: str, testset: list[QAItem] = TESTSET):
    """Runs faithfulness, answer_relevancy, context_precision, and
    context_recall against `collection_name`, judged by the local Ollama
    model (not GPT-4) -- see README for why absolute scores under a local
    judge aren't comparable to published GPT-4-judged benchmarks, only the
    delta between two runs under the *same* judge is meaningful.
    """
    dataset = build_dataset(collection_name, testset)

    judge_llm = LangchainLLMWrapper(get_llm(temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=2, timeout=180),
        raise_exceptions=False,
    )
