"""Multi-stage retrieval pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from seraph.config import EMBEDDERS, SeraphConfig
from seraph.protocols import ChunkStore, Embedder, Reranker, Retriever
from seraph.retrieval.fusion import fuse
from seraph.retrieval.lexical import LexicalRetriever
from seraph.types import AnalyzedQuery, Chunk, ScoredChunk


class Pipeline:
    def __init__(
        self,
        cfg: SeraphConfig,
        store: ChunkStore,
        version: str = "HEAD",
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.version = version
        self._embedder = embedder
        self._reranker = reranker
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
            inner = load_embedder(r.dense_model, self.cfg.device, self.cfg.cache_path, r.dense_fallback)
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

    def retrieve_views(
        self, queries: Sequence[AnalyzedQuery], depth: int
    ) -> dict[str, list[list[ScoredChunk]]]:
        return {name: v.search_batch(queries, depth, self.version) for name, v in self.views.items()}

    def weights_for(self, q: AnalyzedQuery) -> dict[str, float]:
        r = self.cfg.retrieval
        w = q.weights if (r.fusion == "adaptive" and q.weights) else r.static_weights
        return {v: w.get(v, 0.0) for v in self.views}

    def fuse(self, queries: Sequence[AnalyzedQuery], per_view: dict[str, list[list[ScoredChunk]]]) -> list[list[ScoredChunk]]:
        r = self.cfg.retrieval
        out = []
        for i, q in enumerate(queries):
            runs = {v: hits[i] for v, hits in per_view.items()}
            weights = None if r.fusion == "rrf" else self.weights_for(q)
            out.append(fuse(runs, r.fusion, weights, r.rrf_k, r.fusion_norm))
        return out

    @property
    def reranker(self):
        if self._reranker is None:
            from seraph.retrieval.reranker import load_reranker

            r = self.cfg.retrieval
            self._reranker = load_reranker(
                r.reranker_model, self.cfg.device, self.cfg.cache_path, r.rerank_blend, r.reranker_fallback
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
        r = self.cfg.retrieval
        depth = max(k, r.first_stage_k)
        per_view = self.retrieve_views(queries, depth)
        results = self.fuse(queries, per_view)
        if r.use_reranker:
            results = self.rerank(queries, results)
        return [h[:k] for h in results]
