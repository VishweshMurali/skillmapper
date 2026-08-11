import asyncio

from skillmapper.retrieval.cache import EmbeddingCache
from skillmapper.retrieval.embeddings import Embedder


def test_cache_miss_then_hit(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache", "test-model")
    assert cache.get("hello", "document") is None
    assert cache.misses == 1

    cache.set("hello", "document", [1.0, 2.0])
    assert cache.get("hello", "document") == [1.0, 2.0]
    assert cache.hits == 1


def test_cache_discriminates_query_vs_document_for_same_text(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache", "test-model")
    cache.set("same text", "document", [1.0])
    cache.set("same text", "query", [2.0])
    assert cache.get("same text", "document") == [1.0]
    assert cache.get("same text", "query") == [2.0]


class _CountingFakeClient:
    def __init__(self):
        self.calls: list[str] = []

    def get_text_embedding(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text))]


def test_embedder_applies_prefix_and_uses_cache_on_repeat(tmp_path, monkeypatch):
    cache = EmbeddingCache(tmp_path / "cache", "test-model")
    embedder = Embedder(model_name="test-model", cache=cache)
    fake_client = _CountingFakeClient()
    embedder._client = fake_client

    v1 = embedder.embed("hello", "document")
    v2 = embedder.embed("hello", "document")  # should hit cache, not call client again

    assert v1 == v2
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0] == "search_document: hello"


def test_aembed_batch_dedupes_repeated_text_via_cache(tmp_path):
    # concurrency=1 forces strict serialization so the second "a" is
    # guaranteed to see the first "a"'s cache write -- with higher
    # concurrency two concurrent misses for the same text can legitimately
    # both hit the client before either has cached, which is not a bug.
    cache = EmbeddingCache(tmp_path / "cache", "test-model")
    embedder = Embedder(model_name="test-model", cache=cache)
    fake_client = _CountingFakeClient()
    embedder._client = fake_client

    texts = ["a", "b", "a", "c"]
    results = asyncio.run(embedder.aembed_batch(texts, "query", concurrency=1))

    assert len(results) == 4
    # "a" appears twice but should only hit the (fake) client once
    assert len(fake_client.calls) == 3
