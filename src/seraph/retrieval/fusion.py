"""Score fusion across retrieval views. Fused hits keep every view's raw score."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from seraph.types import Chunk, ScoredChunk

ViewHits = Mapping[str, Sequence[ScoredChunk]]
Norm = Literal["minmax", "zscore"]


def _collect(runs: ViewHits) -> tuple[dict[str, Chunk], dict[str, dict[str, float]]]:
    chunks: dict[str, Chunk] = {}
    raw: dict[str, dict[str, float]] = {}
    for view, hits in runs.items():
        for h in hits:
            chunks.setdefault(h.chunk_hash, h.chunk)
            raw.setdefault(h.chunk_hash, {})
            raw[h.chunk_hash][view] = h.scores.get(view, h.score)
            for other, s in h.scores.items():
                raw[h.chunk_hash].setdefault(other, s)
    return chunks, raw


def _emit(chunks: dict[str, Chunk], raw: dict[str, dict[str, float]], fused: dict[str, float]) -> list[ScoredChunk]:
    out = [ScoredChunk(chunks[h], s, dict(raw[h])) for h, s in fused.items()]
    out.sort(key=lambda x: (-x.score, x.chunk_hash))
    return out


def rrf(runs: ViewHits, k: int = 60, weights: Mapping[str, float] | None = None) -> list[ScoredChunk]:
    chunks, raw = _collect(runs)
    fused: dict[str, float] = dict.fromkeys(chunks, 0.0)
    for view, hits in runs.items():
        w = 1.0 if weights is None else weights.get(view, 0.0)
        if w == 0:
            continue
        for rank, h in enumerate(hits):
            fused[h.chunk_hash] += w / (k + rank + 1)
    return _emit(chunks, raw, fused)


def normalize_scores(scores: np.ndarray, norm: Norm) -> np.ndarray:
    if len(scores) == 0:
        return scores
    if norm == "minmax":
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo) if hi > lo else np.ones_like(scores)
    sd = scores.std()
    return (scores - scores.mean()) / sd if sd > 0 else np.zeros_like(scores)


def weighted(runs: ViewHits, weights: Mapping[str, float], norm: Norm = "minmax") -> list[ScoredChunk]:
    """Weighted sum of per-query normalized scores; a doc missing from a view gets that view's floor."""
    chunks, raw = _collect(runs)
    fused: dict[str, float] = dict.fromkeys(chunks, 0.0)
    for view, hits in runs.items():
        w = weights.get(view, 0.0)
        if w == 0 or not hits:
            continue
        vals = normalize_scores(np.array([h.scores.get(view, h.score) for h in hits], dtype=np.float64), norm)
        floor = float(vals.min())
        seen = {h.chunk_hash: float(v) for h, v in zip(hits, vals, strict=True)}
        for d in fused:
            fused[d] += w * seen.get(d, floor)
    return _emit(chunks, raw, fused)


def fuse(
    runs: ViewHits,
    method: str,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    norm: Norm = "minmax",
) -> list[ScoredChunk]:
    runs = {v: h for v, h in runs.items() if h is not None}
    if method == "none" or len(runs) == 1:
        if len(runs) != 1:
            raise ValueError("fusion 'none' needs exactly one view")
        return list(next(iter(runs.values())))
    if method == "rrf":
        return rrf(runs, rrf_k, weights)
    if method in ("weighted", "adaptive"):
        if weights is None:
            raise ValueError(f"{method} fusion needs weights")
        return weighted(runs, weights, norm)
    raise ValueError(f"unknown fusion {method!r}")
