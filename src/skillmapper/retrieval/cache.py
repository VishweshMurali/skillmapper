from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import diskcache

EmbeddingKind = Literal["query", "document"]


class EmbeddingCache:
    """Content-hash-keyed, disk-persisted embedding cache.

    Keyed on (model, kind, sha256(text)) -- `kind` is included because
    nomic-embed-text embeds the same raw text differently depending on
    whether it's used as a query or a document (see embeddings.py), so the
    two must never collide in the cache.
    """

    def __init__(self, cache_dir: Path, model_name: str):
        self._cache = diskcache.Cache(str(cache_dir))
        self._model_name = model_name
        self.hits = 0
        self.misses = 0

    def _key(self, text: str, kind: EmbeddingKind) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._model_name}:{kind}:{digest}"

    def get(self, text: str, kind: EmbeddingKind) -> list[float] | None:
        value = self._cache.get(self._key(text, kind))
        if value is not None:
            self.hits += 1
        else:
            self.misses += 1
        return value

    def set(self, text: str, kind: EmbeddingKind, embedding: list[float]) -> None:
        self._cache.set(self._key(text, kind), embedding)

    def close(self) -> None:
        self._cache.close()
