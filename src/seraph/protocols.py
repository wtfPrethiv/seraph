"""Cross-owner contracts. Changing a protocol needs review from both owners."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from seraph.types import (
    AnalyzedQuery,
    ChangeSet,
    Chunk,
    Direction,
    Edge,
    EdgeKind,
    LineageNode,
    ScoredChunk,
    SymbolDiff,
    Version,
)


@runtime_checkable
class ChunkStore(Protocol):
    def iter_chunks(self, version: str = "HEAD") -> Iterator[Chunk]: ...
    def get(self, chunk_hash: str) -> Chunk: ...
    def changed_since(self, version: str) -> ChangeSet: ...


@runtime_checkable
class CodeGraph(Protocol):
    def neighbors(
        self, symbol_id: str, kinds: set[EdgeKind], direction: Direction | str
    ) -> list[Edge]: ...
    def symbol_for_chunk(self, chunk_hash: str) -> str | None: ...
    def chunk_for_symbol(self, symbol_id: str) -> str | None: ...
    def resolve(self, name: str) -> list[str]: ...


@runtime_checkable
class VersionStore(Protocol):
    def versions(self) -> list[Version]: ...
    def lineage(self, lineage_id: str) -> list[LineageNode]: ...
    def diff_symbol(self, symbol: str, a: str, b: str) -> SymbolDiff: ...


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(self, chunks: Iterable[Chunk]) -> None: ...
    def search(self, query: AnalyzedQuery, k: int, version: str = "HEAD") -> list[ScoredChunk]: ...
    def search_batch(
        self, queries: Sequence[AnalyzedQuery], k: int, version: str = "HEAD"
    ) -> list[list[ScoredChunk]]: ...


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class QueryAnalyzer(Protocol):
    def analyze(self, query: str) -> AnalyzedQuery: ...
