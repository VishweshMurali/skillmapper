"""Ingest the synthetic resume/JD corpus into ChromaDB + BM25.

Usage:
    uv run scripts/ingest.py                 # both variants (production + naive comparison)
    uv run scripts/ingest.py --variant meta   # metadata-aware only
    uv run scripts/ingest.py --variant naive  # naive only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmapper.ingestion.chunking import chunk_document
from skillmapper.ingestion.naive_chunking import chunk_document_naive
from skillmapper.ingestion.pipeline import ingest_sync
from skillmapper.retrieval.vector_store import METADATA_COLLECTION, NAIVE_COLLECTION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["meta", "naive", "both"], default="both")
    args = parser.parse_args()

    if args.variant in ("meta", "both"):
        start = time.perf_counter()
        chunks = ingest_sync(METADATA_COLLECTION, chunk_document)
        print(f"[meta]  ingested {len(chunks)} chunks into '{METADATA_COLLECTION}' "
              f"in {time.perf_counter() - start:.1f}s")

    if args.variant in ("naive", "both"):
        start = time.perf_counter()
        chunks = ingest_sync(NAIVE_COLLECTION, chunk_document_naive)
        print(f"[naive] ingested {len(chunks)} chunks into '{NAIVE_COLLECTION}' "
              f"in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
