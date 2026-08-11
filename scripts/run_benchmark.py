"""Real, measured before/after numbers for the caching + async-parallelization
optimization story. Requires `scripts/ingest.py` to have been run first
(both collections must exist for the storage comparison).

Usage:
    uv run scripts/run_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmapper.config import settings
from skillmapper.eval.benchmark import run_all_benchmarks


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def main() -> None:
    results = run_all_benchmarks()

    cache = results["cache"]
    print(f"\n=== Embedding cache ({cache['n_texts']} texts) ===")
    print(f"cold (all misses): {cache['cold_seconds']:.2f}s")
    print(f"warm (all hits):   {cache['warm_seconds']:.2f}s")
    print(f"speedup:           {cache['speedup']:.1f}x")

    par = results["parallelism"]
    print(f"\n=== Async parallel embedding ({par['n_texts']} texts, concurrency={par['concurrency']}) ===")
    print(f"serial:   {par['serial_seconds']:.2f}s")
    print(f"parallel: {par['parallel_seconds']:.2f}s")
    print(f"speedup:  {par['speedup']:.1f}x")

    print("\n=== Storage (this repo's demo corpus) ===")
    for name, stats in results["storage"].items():
        print(
            f"{name:28s} chunks={stats['n_chunks']:3d}  "
            f"text={_fmt_bytes(stats['text_bytes']):>8s}  "
            f"embeddings={_fmt_bytes(stats['embedding_bytes']):>8s}  "
            f"total={_fmt_bytes(stats['total_bytes']):>8s}"
        )

    out_path = settings.cache_dir / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
