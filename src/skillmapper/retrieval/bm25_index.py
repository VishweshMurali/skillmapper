from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Wraps rank_bm25.BM25Okapi with the entry metadata needed to filter
    and re-hydrate results. rank_bm25 has no native metadata filtering, so
    callers (hybrid.py) must post-filter `search()` results themselves.
    """

    def __init__(self, entries: list[dict[str, Any]]):
        self.entries = entries
        self._bm25 = BM25Okapi([tokenize(e["text"]) for e in entries]) if entries else None

    def search(self, query: str) -> list[tuple[dict[str, Any], float]]:
        """Score every entry in the corpus against `query`, best first."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.entries, scores), key=lambda pair: pair[1], reverse=True)
        return ranked

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.entries, f)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with open(path, "rb") as f:
            entries = pickle.load(f)
        return cls(entries)

    @classmethod
    def build_from_collection(cls, entries: list[dict[str, Any]]) -> BM25Index:
        return cls(entries)
