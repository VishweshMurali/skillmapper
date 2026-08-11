from __future__ import annotations

import asyncio

from llama_index.embeddings.ollama import OllamaEmbedding

from skillmapper.config import settings
from skillmapper.retrieval.cache import EmbeddingCache, EmbeddingKind

# nomic-embed-text is trained with task-instruction prefixes; skipping them
# measurably degrades retrieval quality. OllamaEmbedding does not add these
# automatically, so we apply them explicitly here and nowhere else.
_PREFIXES: dict[EmbeddingKind, str] = {
    "query": "search_query: ",
    "document": "search_document: ",
}

DEFAULT_CONCURRENCY = 4


class Embedder:
    def __init__(self, model_name: str | None = None, cache: EmbeddingCache | None = None):
        self.model_name = model_name or settings.ollama_embed_model
        self._client = OllamaEmbedding(
            model_name=self.model_name, base_url=settings.ollama_base_url
        )
        self._cache = cache or EmbeddingCache(settings.cache_dir / "embeddings", self.model_name)

    def embed(self, text: str, kind: EmbeddingKind) -> list[float]:
        cached = self._cache.get(text, kind)
        if cached is not None:
            return cached
        prefixed = _PREFIXES[kind] + text
        vector = self._client.get_text_embedding(prefixed)
        self._cache.set(text, kind, vector)
        return vector

    async def aembed_batch(
        self, texts: list[str], kind: EmbeddingKind, concurrency: int = DEFAULT_CONCURRENCY
    ) -> list[list[float]]:
        """Embed many texts concurrently. Cache hits resolve immediately and
        never touch the semaphore; only cache misses hit Ollama, bounded by
        `concurrency` in-flight requests at a time.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(text: str) -> list[float]:
            cached = self._cache.get(text, kind)
            if cached is not None:
                return cached
            async with semaphore:
                return await asyncio.to_thread(self.embed, text, kind)

        return await asyncio.gather(*(_one(t) for t in texts))
