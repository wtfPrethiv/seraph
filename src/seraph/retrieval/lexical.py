"""BM25 view (alpha) over code-aware tokens."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from pathlib import Path

import bm25s
import numpy as np

from seraph.retrieval.base import BaseRetriever
from seraph.retrieval.tokenize import tokenize
from seraph.types import AnalyzedQuery, Chunk, ScoredChunk, View


class LexicalRetriever(BaseRetriever):
    name = View.LEXICAL.value

    def __init__(self, k1: float = 1.2, b: float = 0.75, stem: bool = True) -> None:
        self.k1, self.b, self.stem = k1, b, stem
        self._bm25: bm25s.BM25 | None = None
        self._chunks: list[Chunk] = []

    def index(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        corpus_tokens = [tokenize(c.text, self.stem) or [""] for c in self._chunks]
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self._bm25.index(corpus_tokens, show_progress=False)

    def query_tokens(self, q: AnalyzedQuery) -> list[str]:
        assert self._bm25 is not None
        vocab = self._bm25.vocab_dict
        return [t for t in tokenize(q.text, self.stem) if t and t in vocab]

    def search_batch(
        self, queries: Sequence[AnalyzedQuery], k: int, version: str = "HEAD"
    ) -> list[list[ScoredChunk]]:
        assert self._bm25 is not None, "index() first"
        k = min(k, len(self._chunks))
        out: list[list[ScoredChunk]] = [[] for _ in queries]
        toks = [self.query_tokens(q) for q in queries]
        live = [i for i, t in enumerate(toks) if t]
        if not live:
            return out
        idx, scores = self._bm25.retrieve([toks[i] for i in live], k=k, show_progress=False)
        for qi, row_i, row_s in zip(live, idx, scores, strict=True):
            out[qi] = [
                ScoredChunk(self._chunks[i], float(s), {self.name: float(s)})
                for i, s in zip(row_i, row_s, strict=True)
                if s > 0
            ]
        return out

    def score_all(self, query: AnalyzedQuery) -> np.ndarray:
        """Dense score vector over the whole corpus (used by adaptive fusion features)."""
        assert self._bm25 is not None
        return np.asarray(self._bm25.get_scores(self.query_tokens(query)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str | Path) -> LexicalRetriever:
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
