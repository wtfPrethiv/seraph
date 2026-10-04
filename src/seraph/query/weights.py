"""Per-query view weights (alpha lexical, beta semantic, gamma structural, delta graph, epsilon evolution).

`RuleWeights` maps the query type and parser features to weights. `LearnedWeights` is trained on
per-query oracle utilities: every view's run is cached once, each query's NDCG@10 is computed for
every cell of a weight simplex grid, and a regressor per cell predicts that utility from query and
first-stage score features; the predicted-best cell is used at query time.
"""

from __future__ import annotations

import itertools
import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from seraph.retrieval.fusion import normalize_scores
from seraph.types import AnalyzedQuery, QueryType, ScoredChunk

ViewRuns = Mapping[str, Sequence[ScoredChunk]]

TYPE_PRIORS: dict[QueryType, dict[str, float]] = {
    QueryType.LEXICAL: {"lexical": 0.55, "semantic": 0.35, "structural": 0.1},
    QueryType.SEMANTIC: {"lexical": 0.15, "semantic": 0.7, "structural": 0.15},
    QueryType.STRUCTURAL: {"lexical": 0.15, "semantic": 0.6, "structural": 0.25},
    QueryType.DEPENDENCY: {"lexical": 0.25, "semantic": 0.3, "graph": 0.45},
    QueryType.EVOLUTIONARY: {"lexical": 0.2, "semantic": 0.3, "evolution": 0.5},
    QueryType.MIXED: {"lexical": 0.25, "semantic": 0.5, "structural": 0.1, "graph": 0.075, "evolution": 0.075},
}


def _ensure_parsed(q: AnalyzedQuery) -> AnalyzedQuery:
    if q.features and q.type_probs:
        return q
    from seraph.query.analyzer import RuleQueryAnalyzer

    aq = RuleQueryAnalyzer().analyze(q.raw or q.text)
    aq.qid, aq.text, aq.sub_queries = q.qid, q.text, q.sub_queries
    return aq


class RuleWeights:
    """Type-prior mixture (by classifier probabilities) nudged by identifier density and length."""

    def weights(self, q: AnalyzedQuery, views: Sequence[str] | None = None) -> dict[str, float]:
        q = _ensure_parsed(q)
        probs = q.type_probs or {q.query_type.value: 1.0}
        w: dict[str, float] = {}
        for t, p in probs.items():
            for v, x in TYPE_PRIORS[QueryType(t)].items():
                w[v] = w.get(v, 0.0) + p * x
        f = q.features
        ident = f.get("identifier_ratio", 0.0)
        w["lexical"] = w.get("lexical", 0.0) + 0.4 * ident
        if f.get("log_len", 0.0) > math.log(60):
            w["semantic"] = w.get("semantic", 0.0) + 0.1
        if f.get("has_io_sections", 0.0):
            w["structural"] = w.get("structural", 0.0) + 0.05
        if views is not None:
            w = {v: w.get(v, 0.0) for v in views}
        total = sum(w.values()) or 1.0
        return {v: x / total for v, x in w.items()}

    def weights_batch(self, queries: Sequence[AnalyzedQuery], per_view: Mapping[str, list]) -> list[dict[str, float]]:
        views = list(per_view)
        return [self.weights(q, views) for q in queries]


def simplex_grid(views: Sequence[str], step: float = 0.1) -> np.ndarray:
    n = round(1 / step)
    cells = [c for c in itertools.product(range(n + 1), repeat=len(views)) if sum(c) == n]
    return np.array(cells, dtype=np.float64) / n


def view_matrix(runs: ViewRuns, views: Sequence[str], norm: str = "minmax") -> tuple[list[str], np.ndarray]:
    """Candidates x views matrix of normalized scores, matching `fusion.weighted` (missing -> floor)."""
    hashes: dict[str, int] = {}
    for v in views:
        for h in runs.get(v, ()):
            hashes.setdefault(h.chunk_hash, len(hashes))
    m = np.zeros((len(hashes), len(views)))
    for j, v in enumerate(views):
        hits = runs.get(v) or []
        if not hits:
            continue
        vals = normalize_scores(np.array([h.scores.get(v, h.score) for h in hits], dtype=np.float64), norm)
        m[:, j] = vals.min()
        for h, x in zip(hits, vals, strict=True):
            m[hashes[h.chunk_hash], j] = x
    return list(hashes), m


def cell_utilities(
    runs: Sequence[ViewRuns], qrels: Sequence[Mapping[str, int]], views: Sequence[str], grid: np.ndarray,
    norm: str = "minmax", k: int = 10,
) -> np.ndarray:
    """NDCG@k of every grid cell for every query: shape (n_queries, n_cells)."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    out = np.zeros((len(runs), len(grid)))
    for i, (run, rel) in enumerate(zip(runs, qrels, strict=True)):
        ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
        idcg = float(np.dot(ideal, discounts[: len(ideal)]))
        hashes, m = view_matrix(run, views, norm)
        if idcg == 0 or not hashes:
            continue
        gains = np.array([rel.get(h, 0) for h in hashes], dtype=np.float64)
        if not gains.any():
            continue
        order = np.argsort(np.array(hashes))
        fused = m[order] @ grid.T
        g = gains[order]
        top = np.argsort(-fused, axis=0, kind="stable")[:k]
        out[i] = (g[top] * discounts[: len(top), None]).sum(axis=0) / idcg
    return out


def _stats(hits: Sequence[ScoredChunk], view: str) -> list[float]:
    s = np.array([h.scores.get(view, h.score) for h in hits[:10]], dtype=np.float64)
    if len(s) == 0:
        return [0.0] * 6
    top2 = s[1] if len(s) > 1 else s[0]
    z = normalize_scores(s, "minmax")
    p = np.exp(z * 5)
    p /= p.sum()
    entropy = float(-(p * np.log(p)).sum() / math.log(max(len(p), 2)))
    spread = s[0] - s[-1]
    return [float(s[0]), float((s[0] - top2) / (abs(spread) + 1e-9)), float(s.mean()), float(s.std()), entropy, float(len(hits))]


def query_features(q: AnalyzedQuery, runs: ViewRuns, views: Sequence[str]) -> np.ndarray:
    q = _ensure_parsed(q)
    f = [q.features[k] for k in sorted(q.features)]
    f += [q.type_probs.get(t.value, 0.0) for t in QueryType]
    for v in views:
        f += _stats(runs.get(v) or [], v)
    for a, b in itertools.combinations(views, 2):
        ta = {h.chunk_hash for h in (runs.get(a) or [])[:10]}
        tb = {h.chunk_hash for h in (runs.get(b) or [])[:10]}
        f.append(len(ta & tb) / 10.0)
    return np.array(f, dtype=np.float64)


class LearnedWeights:
    """Predict per-cell utility, use the argmax cell. `kind` is 'ridge' or 'lgbm'."""

    def __init__(
        self, views: Sequence[str], step: float = 0.1, kind: str = "lgbm", norm: str = "minmax", top_cells: int = 0
    ) -> None:
        self.views = list(views)
        self.grid = simplex_grid(self.views, step)
        self.kind = kind
        self.norm = norm
        self.top_cells = top_cells
        self.cells = np.arange(len(self.grid))
        self.models: list = []
        self.fallback = 0

    def fit(self, x: np.ndarray, utility: np.ndarray) -> LearnedWeights:
        """`top_cells` > 0 restricts the choice to the cells with the best mean training utility."""
        mean = utility.mean(axis=0)
        self.fallback = int(mean.argmax())
        if self.top_cells:
            self.cells = np.argsort(-mean, kind="stable")[: self.top_cells]
            utility = utility[:, self.cells]
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            self.models = [make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(x, utility)]
        else:
            import lightgbm as lgb

            params = dict(n_estimators=80, learning_rate=0.05, num_leaves=7, min_child_samples=30,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
            self.models = [lgb.LGBMRegressor(**params).fit(x, utility[:, j]) for j in range(utility.shape[1])]
        return self

    def predict_cells(self, x: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.full(len(x), self.fallback)
        if self.kind == "ridge":
            pred = self.models[0].predict(x).reshape(len(x), -1)
        else:
            pred = np.stack([m.predict(x) for m in self.models], axis=1)
        return self.cells[pred.argmax(axis=1)]

    def weights_batch(self, queries: Sequence[AnalyzedQuery], per_view: Mapping[str, list]) -> list[dict[str, float]]:
        runs = [{v: per_view[v][i] for v in self.views if v in per_view} for i in range(len(queries))]
        x = np.stack([query_features(q, r, self.views) for q, r in zip(queries, runs, strict=True)])
        return [dict(zip(self.views, self.grid[c].tolist(), strict=True)) for c in self.predict_cells(x)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> LearnedWeights:
        with open(path, "rb") as f:
            return pickle.load(f)


DEFAULT_MODEL_PATH = "experiments/models/adaptive_weights.pkl"
