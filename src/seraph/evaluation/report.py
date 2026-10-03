"""Render experiment result JSONs as markdown tables."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

DEFAULT_COLS = ("ndcg@10", "mrr@10", "recall@10", "recall@100", "latency_p50_ms")


def load_results(results_dir: str | Path, prefix: str = "") -> list[dict]:
    out = []
    for p in sorted(Path(results_dir).glob(f"{prefix}*.json")):
        out.append(json.loads(p.read_text()))
    return out


def table(
    results: Sequence[dict],
    splits: Sequence[str] = ("dev", "dev_stdin"),
    cols: Sequence[str] = DEFAULT_COLS,
    order: Sequence[str] | None = None,
) -> str:
    by_name: dict[str, dict[str, dict]] = {}
    for r in results:
        by_name.setdefault(r["name"], {})[r["split"]] = r
    names = [n for n in (order or sorted(by_name)) if n in by_name]
    header = ["config"] + [f"{s} {c}" for s in splits for c in cols]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for n in names:
        row = [n]
        for s in splits:
            m = by_name[n].get(s, {}).get("metrics", {})
            for c in cols:
                v = m.get(c)
                row.append("-" if v is None else (f"{v:.1f}" if c.startswith("latency") else f"{v:.4f}"))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
