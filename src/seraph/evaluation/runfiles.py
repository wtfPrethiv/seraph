"""TREC run files: `qid Q0 docid rank score tag`."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from seraph.evaluation.metrics import Run, ranked


def write_run(run: Run, path: str | Path, tag: str = "seraph", k: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(run):
            docs = run[qid]
            for rank, d in enumerate(ranked(docs, k), start=1):
                f.write(f"{qid} Q0 {d} {rank} {docs[d]:.6f} {tag}\n")


def read_run(path: str | Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = defaultdict(dict)
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            qid, _, d, _, score, *_ = line.split()
            run[qid][d] = float(score)
    return dict(run)
