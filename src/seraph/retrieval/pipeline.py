"""Multi-stage retrieval pipeline."""

from __future__ import annotations

import time
from collections.abc import Sequence

from seraph.config import EMBEDDERS, SeraphConfig
from seraph.protocols import ChunkStore, CodeGraph, Embedder, Reranker, Retriever, VersionStore
from seraph.retrieval.fusion import fuse, rrf
from seraph.retrieval.lexical import LexicalRetriever
from seraph.types import AnalyzedQuery, Chunk, ScoredChunk, SearchResponse, SearchResult, SubQuery


class Pipeline:
    def __init__(
        self,
        cfg: SeraphConfig,
        store: ChunkStore,
        version: str = "HEAD",
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        graph: CodeGraph | None = None,
        versions: VersionStore | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.version = version
        self.graph = graph
        self.versions = versions
        self._embedder = embedder
        self._reranker = reranker
        self._analyzer = None
        self.views: dict[str, Retriever] = {}

    @classmethod
    def from_config(cls, cfg: SeraphConfig, store: ChunkStore, version: str = "HEAD", **kw) -> Pipeline:
        p = cls(cfg, store, version, **kw)
        p.build()
        return p

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            from seraph.retrieval.semantic import CachedEmbedder, load_embedder

            r = self.cfg.retrieval
            inner = load_embedder(
                r.dense_model, self.cfg.device, self.cfg.cache_path, r.dense_fallback, self.cfg.allow_local_models
            )
            if EMBEDDERS[inner.name].is_api:
                self._embedder = inner  # API backend caches per text itself
            else:
                self._embedder = CachedEmbedder(
                    inner, self.cfg.cache_path, EMBEDDERS[inner.name], self.cfg.cache_query_embeddings
                )
        return self._embedder

    def build(self) -> None:
        r = self.cfg.retrieval
        chunks: list[Chunk] = list(self.store.iter_chunks(self.version))
        if r.use_bm25:
            lex = LexicalRetriever(r.bm25_k1, r.bm25_b, r.bm25_stem)
            lex.index(chunks)
            self.views[lex.name] = lex
        if r.use_dense:
            from seraph.retrieval.semantic import DenseRetriever

            dense = DenseRetriever(self.embedder)
            dense.index(chunks)
            self.views[dense.name] = dense

    def model_metadata(self) -> dict[str, str]:
        meta = {name: getattr(v, "revision", name) for name, v in self.views.items()}
        if self._reranker is not None:
            meta["rerank"] = f"{self._reranker.name}@{getattr(self._reranker, 'revision', 'unknown')}"
        return meta

    @property
    def analyzer(self):
        if self._analyzer is None:
            from seraph.query.analyzer import build_analyzer

            self._analyzer = build_analyzer(self.cfg, self.weighter)
        return self._analyzer

    @property
    def weighter(self):
        return None

    def analyze(self, queries: Sequence[AnalyzedQuery]) -> list[AnalyzedQuery]:
        if not self.cfg.retrieval.use_query_analyzer:
            return list(queries)
        out = []
        for q in queries:
            aq = self.analyzer.analyze(q.raw)
            aq.qid = q.qid
            out.append(aq)
        return out

    def retrieve_views(
        self, queries: Sequence[AnalyzedQuery], depth: int
    ) -> dict[str, list[list[ScoredChunk]]]:
        if not any(len(q.sub_queries) > 1 for q in queries):
            return {name: v.search_batch(queries, depth, self.version) for name, v in self.views.items()}
        flat: list[AnalyzedQuery] = []
        owner: list[tuple[int, float]] = []
        for i, q in enumerate(queries):
            for s in q.sub_queries or [SubQuery(q.text)]:
                flat.append(AnalyzedQuery(raw=q.raw, text=s.text, query_type=q.query_type, qid=q.qid))
                owner.append((i, s.weight))
        out: dict[str, list[list[ScoredChunk]]] = {}
        for name, v in self.views.items():
            flat_hits = v.search_batch(flat, depth, self.version)
            grouped: list[dict[str, list[ScoredChunk]]] = [{} for _ in queries]
            weights: list[dict[str, float]] = [{} for _ in queries]
            for j, ((qi, w), hits) in enumerate(zip(owner, flat_hits, strict=True)):
                key = f"sub{j}"
                grouped[qi][key] = [ScoredChunk(h.chunk, h.score, {key: h.score}) for h in hits]
                weights[qi][key] = w
            combined = []
            for g, w in zip(grouped, weights, strict=True):
                hits = rrf(g, self.cfg.retrieval.rrf_k, w)[:depth]
                combined.append([ScoredChunk(h.chunk, h.score, {name: h.score}) for h in hits])
            out[name] = combined
        return out

    def weights_for(self, q: AnalyzedQuery, views: Sequence[str] | None = None) -> dict[str, float]:
        r = self.cfg.retrieval
        w = q.weights if (r.fusion == "adaptive" and q.weights) else r.static_weights
        return {v: w.get(v, 0.0) for v in (views or self.views)}

    def graph_view(
        self, queries: Sequence[AnalyzedQuery], seeds: list[list[ScoredChunk]]
    ) -> list[list[ScoredChunk]]:
        from seraph.graph.traversal import direction_for, expand, kinds_for

        r = self.cfg.retrieval
        assert self.graph is not None
        return [
            expand(s, self.graph, self.store, kinds_for(q), direction_for(q), r.graph_hops, r.graph_decay)
            for q, s in zip(queries, seeds, strict=True)
        ]

    def fuse(self, queries: Sequence[AnalyzedQuery], per_view: dict[str, list[list[ScoredChunk]]]) -> list[list[ScoredChunk]]:
        r = self.cfg.retrieval
        out = []
        for i, q in enumerate(queries):
            runs = {v: hits[i] for v, hits in per_view.items()}
            weights = None if r.fusion == "rrf" else self.weights_for(q, list(runs))
            out.append(fuse(runs, r.fusion, weights, r.rrf_k, r.fusion_norm))
        return out

    @property
    def reranker(self):
        if self._reranker is None:
            from seraph.retrieval.reranker import load_reranker

            r = self.cfg.retrieval
            self._reranker = load_reranker(
                r.reranker_model,
                self.cfg.device,
                self.cfg.cache_path,
                r.rerank_blend,
                r.reranker_fallback,
                self.cfg.allow_local_models,
            )
        return self._reranker

    def rerank(self, queries: Sequence[AnalyzedQuery], fused: list[list[ScoredChunk]]) -> list[list[ScoredChunk]]:
        depth = self.cfg.retrieval.rerank_depth
        heads = self.reranker.rerank_batch([q.text for q in queries], [f[:depth] for f in fused], depth)
        out = []
        for head, f in zip(heads, fused, strict=True):
            floor = min((h.score for h in head), default=0.0)
            tail = [ScoredChunk(t.chunk, floor - (j + 1) * 1e-3, t.scores) for j, t in enumerate(f[depth:])]
            out.append(head + tail)
        return out

    def search_batch(self, queries: Sequence[AnalyzedQuery], k: int) -> list[list[ScoredChunk]]:
        return self._search_analyzed(self.analyze(queries), k)

    def _search_analyzed(self, queries: Sequence[AnalyzedQuery], k: int) -> list[list[ScoredChunk]]:
        r = self.cfg.retrieval
        depth = max(k, r.first_stage_k)
        per_view = self.retrieve_views(queries, depth)
        if r.use_graph_expansion and self.graph is not None:
            per_view["graph"] = self.graph_view(queries, self.fuse(queries, per_view))
        results = self.fuse(queries, per_view)
        if r.use_reranker:
            results = self.rerank(queries, results)
        return [h[:k] for h in results]

    def search(
        self, query: str, top_k: int = 10, version: str | None = None, include_history: bool = False
    ) -> SearchResponse:
        t0 = time.perf_counter()
        (aq,) = self.analyze([AnalyzedQuery.plain(query)])
        hits = self._search_analyzed([aq], top_k)[0]
        return SearchResponse(
            query=query,
            query_type=aq.query_type.value,
            version=version or self.version,
            results=[to_result(h) for h in hits],
            sub_queries=[s.text for s in aq.sub_queries],
            weights=self.weights_for(aq) if self.cfg.retrieval.fusion != "rrf" else {},
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def to_result(h: ScoredChunk, snippet_lines: int = 40) -> SearchResult:
    c = h.chunk
    return SearchResult(
        chunk_hash=c.chunk_hash,
        file=c.file,
        symbol=c.symbol,
        kind=c.kind,
        start_line=c.start_line,
        end_line=c.end_line,
        version_id=c.version_id,
        score=float(h.score),
        retrieval_scores={k: float(v) for k, v in h.scores.items()},
        snippet="\n".join(c.text.splitlines()[:snippet_lines]),
        lineage_id=c.lineage_id,
    )


_REPOS: dict[str, Pipeline] = {}


def register_repo(repo: str, pipeline: Pipeline) -> None:
    """Indexing side (Person B) registers a built pipeline per repo; `search()` looks it up."""
    _REPOS[repo] = pipeline


def get_pipeline(repo: str) -> Pipeline:
    try:
        return _REPOS[repo]
    except KeyError:
        raise KeyError(f"repo {repo!r} is not indexed; call register_repo() first") from None


def search(
    query: str,
    repo: str,
    top_k: int = 10,
    version: str = "HEAD",
    include_history: bool = False,
) -> SearchResponse:
    """Public entry point used by the CLI and MCP server."""
    return get_pipeline(repo).search(query, top_k, version, include_history)
