"""Multi-stage retrieval pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from seraph.config import SeraphConfig
from seraph.protocols import ChunkStore, Retriever
from seraph.retrieval.lexical import LexicalRetriever
from seraph.types import AnalyzedQuery, Chunk, ScoredChunk


class Pipeline:
    def __init__(self, cfg: SeraphConfig, store: ChunkStore, version: str = "HEAD") -> None:
        self.cfg = cfg
        self.store = store
        self.version = version
        self.views: dict[str, Retriever] = {}

    @classmethod
    def from_config(cls, cfg: SeraphConfig, store: ChunkStore, version: str = "HEAD") -> Pipeline:
        p = cls(cfg, store, version)
        p.build()
        return p

    def build(self) -> None:
        r = self.cfg.retrieval
        chunks: list[Chunk] = list(self.store.iter_chunks(self.version))
        if r.use_bm25:
            lex = LexicalRetriever(r.bm25_k1, r.bm25_b, r.bm25_stem)
            lex.index(chunks)
            self.views[lex.name] = lex

    def model_metadata(self) -> dict[str, str]:
        return {name: getattr(v, "revision", name) for name, v in self.views.items()}

    def search_batch(self, queries: Sequence[AnalyzedQuery], k: int) -> list[list[ScoredChunk]]:
        depth = max(k, self.cfg.retrieval.first_stage_k)
        (name, view), = self.views.items()
        return [hits[:k] for hits in view.search_batch(queries, depth, self.version)]
