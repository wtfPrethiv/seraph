"""Evolution-aware ranking: lineage dedup, the evolution view (epsilon), and MMR diversification."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from seraph.protocols import VersionStore
from seraph.retrieval.fusion import normalize_scores
from seraph.retrieval.tokenize import tokenize
from seraph.types import AnalyzedQuery, ScoredChunk, View

HEAD = "HEAD"


def version_rank(versions: VersionStore | None) -> dict[str, int]:
    """Oldest -> 0. HEAD ranks after every recorded version."""
    rank = {v.id: i for i, v in enumerate(versions.versions())} if versions else {}
    rank[HEAD] = len(rank) + 1
    return rank


def _group_key(h: ScoredChunk) -> str:
    return h.chunk.lineage_id or f"chunk:{h.chunk_hash}"


def lineage_dedup(
    hits: Sequence[ScoredChunk],
    versions: VersionStore | None = None,
    target_version: str = HEAD,
    keep_history: bool = False,
) -> list[ScoredChunk]:
    """One representative per lineage (the target version if retrieved, else the newest).

    With `keep_history`, every version is kept, grouped by lineage and ordered newest first.
    """
    rank = version_rank(versions)
    groups: dict[str, list[ScoredChunk]] = {}
    for h in hits:
        groups.setdefault(_group_key(h), []).append(h)

    def newest_first(members: list[ScoredChunk]) -> list[ScoredChunk]:
        return sorted(members, key=lambda m: -rank.get(m.chunk.version_id, -1))

    ordered = sorted(groups.values(), key=lambda g: -max(m.score for m in g))
    out: list[ScoredChunk] = []
    for g in ordered:
        best = max(m.score for m in g)
        if keep_history:
            members = newest_first(g)
            out += [ScoredChunk(m.chunk, best - i * 1e-6, dict(m.scores)) for i, m in enumerate(members)]
            continue
        match = [m for m in g if m.chunk.version_id == target_version]
        rep = match[0] if match else newest_first(g)[0]
        out.append(ScoredChunk(rep.chunk, best, dict(rep.scores)))
    return out


def lineage_history(versions: VersionStore | None, lineage_id: str | None) -> list[dict[str, Any]]:
    if versions is None or lineage_id is None:
        return []
    return [
        {
            "version_id": n.version_id,
            "symbol": n.symbol_id,
            "change_type": n.change_type.value,
            "chunk_hash": n.chunk_hash,
            "commit_message": n.commit_message,
        }
        for n in versions.lineage(lineage_id)
    ]


def evolution_view(
    query: AnalyzedQuery, candidates: Sequence[ScoredChunk], versions: VersionStore
) -> list[ScoredChunk]:
    """Score candidates by how well their lineage's commit messages / tags match the query."""
    q_toks = set(tokenize(query.text, stem=True))
    refs = {r.lower() for r in query.version_refs}
    tags = {v.id: (v.tag or "").lower() for v in versions.versions()}
    out = []
    seen: set[str] = set()
    for c in candidates:
        lid = c.chunk.lineage_id
        if not lid or lid in seen:
            continue
        seen.add(lid)
        best = 0.0
        for n in versions.lineage(lid):
            msg = set(tokenize(n.commit_message, stem=True))
            overlap = len(q_toks & msg) / max(len(q_toks), 1)
            ref_hit = 0.5 if refs & {n.version_id.lower(), tags.get(n.version_id, "")} else 0.0
            changed = 0.1 if n.change_type.value not in ("unchanged", "added") else 0.0
            best = max(best, overlap + ref_hit + changed)
        if best > 0:
            out.append(ScoredChunk(c.chunk, best, {View.EVOLUTION.value: best}))
    out.sort(key=lambda x: (-x.score, x.chunk_hash))
    return out


_WORD = re.compile(r"\w+")


def token_jaccard_matrix(texts: Sequence[str]) -> np.ndarray:
    sets = [set(_WORD.findall(t.lower())) for t in texts]
    n = len(sets)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            u = len(sets[i] | sets[j])
            sim[i, j] = sim[j, i] = len(sets[i] & sets[j]) / u if u else 0.0
    return sim


def mmr(
    hits: Sequence[ScoredChunk],
    lam: float = 0.7,
    k: int | None = None,
    vectors: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> list[ScoredChunk]:
    """Maximal Marginal Relevance over the given hits; `vectors` maps chunk hashes to embeddings."""
    hits = list(hits)
    if len(hits) < 3:
        return hits
    k = min(k or len(hits), len(hits))
    raw = np.array([h.score for h in hits], dtype=np.float64)
    rel = raw / raw.max() if raw.min() >= 0 and raw.max() > 0 else normalize_scores(raw, "minmax")
    if vectors is not None:
        v = vectors([h.chunk_hash for h in hits])
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        sim = v @ v.T
    else:
        sim = token_jaccard_matrix([h.chunk.text for h in hits])
    chosen: list[int] = [int(np.argmax(rel))]
    left = set(range(len(hits))) - set(chosen)
    while left and len(chosen) < k:
        idx = list(left)
        red = sim[np.ix_(idx, chosen)].max(axis=1)
        val = lam * rel[idx] - (1 - lam) * red
        pick = idx[int(np.argmax(val))]
        chosen.append(pick)
        left.remove(pick)
    top = hits[chosen[0]].score
    out = [ScoredChunk(hits[i].chunk, top - r * 1e-4, hits[i].scores) for r, i in enumerate(chosen)]
    floor = out[-1].score
    rest = [h for i, h in enumerate(hits) if i not in set(chosen)]
    return out + [ScoredChunk(h.chunk, floor - (j + 1) * 1e-4, h.scores) for j, h in enumerate(rest)]
