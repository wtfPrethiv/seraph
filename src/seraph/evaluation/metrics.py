"""IR metrics. Ranking ties are broken like trec_eval (score desc, then doc id desc)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

Run = Mapping[str, Mapping[str, float]]
Qrels = Mapping[str, Mapping[str, int]]


def ranked(docs: Mapping[str, float], k: int | None = None) -> list[str]:
    order = sorted(docs.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ids = [d for d, _ in order]
    return ids[:k] if k else ids


def ndcg_at_k(ranking: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranking: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    for i, d in enumerate(ranking[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranking: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranking[:k])) / len(relevant)


def evaluate(
    run: Run,
    qrels: Qrels,
    ks: Sequence[int] = (1, 10, 100),
    per_query: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Mean metrics over all queries in qrels (missing queries in run score 0)."""
    pq: dict[str, dict[str, float]] = {}
    for qid, rels in qrels.items():
        if not any(r > 0 for r in rels.values()):
            continue
        ranking = ranked(run.get(qid, {}))
        m: dict[str, float] = {}
        for k in ks:
            m[f"ndcg@{k}"] = ndcg_at_k(ranking, rels, k)
            m[f"mrr@{k}"] = mrr_at_k(ranking, rels, k)
            m[f"recall@{k}"] = recall_at_k(ranking, rels, k)
        pq[qid] = m
    keys = next(iter(pq.values())).keys() if pq else []
    agg = {key: float(np.mean([m[key] for m in pq.values()])) for key in keys}
    return (agg, pq) if per_query else agg


def latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {}
    a = np.asarray(latencies_ms)
    return {
        "latency_p50_ms": float(np.percentile(a, 50)),
        "latency_p95_ms": float(np.percentile(a, 95)),
        "latency_mean_ms": float(a.mean()),
    }


def alpha_ndcg_at_k(
    ranking: Sequence[str],
    doc_intents: Mapping[str, set[str]],
    k: int,
    alpha: float = 0.5,
) -> float:
    """alpha-nDCG (Clarke et al. 2008). `doc_intents` maps relevant doc -> intents it covers."""

    def gain(seq: Sequence[str]) -> float:
        seen: dict[str, int] = {}
        total = 0.0
        for i, d in enumerate(seq[:k]):
            g = 0.0
            for intent in doc_intents.get(d, ()):
                g += (1 - alpha) ** seen.get(intent, 0)
                seen[intent] = seen.get(intent, 0) + 1
            total += g / math.log2(i + 2)
        return total

    # greedy ideal ordering
    pool = list(doc_intents)
    ideal: list[str] = []
    seen: dict[str, int] = {}
    while pool and len(ideal) < k:
        best = max(pool, key=lambda d: sum((1 - alpha) ** seen.get(t, 0) for t in doc_intents[d]))
        ideal.append(best)
        pool.remove(best)
        for t in doc_intents[best]:
            seen[t] = seen.get(t, 0) + 1
    ig = gain(ideal)
    return gain(ranking) / ig if ig > 0 else 0.0


def intra_list_similarity(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity of a result list (lower is more diverse)."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    e = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    sim = e @ e.T
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))
