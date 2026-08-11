"""Run RAGAS against both the naive and metadata-aware chunking collections
and print a markdown comparison table -- this reproduces the debugging
story: naive chunking crosses section boundaries, which should show up as
lower faithfulness/context recall under RAGAS.

Expect this to take 20-60+ minutes on consumer hardware (~12 questions x 4
metrics x 2 collections, each metric potentially multiple local-LLM judge
calls). Writes the result table to data/cache/chunking_comparison.md.

Usage:
    uv run scripts/run_chunking_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmapper.config import settings
from skillmapper.eval.chunking_comparison import run_comparison


def main() -> None:
    table = run_comparison()
    print()
    print(table)

    out_path = settings.cache_dir / "chunking_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
