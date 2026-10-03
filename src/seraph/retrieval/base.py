from __future__ import annotations

from collections.abc import Iterable, Sequence

from seraph.types import AnalyzedQuery, Chunk, ScoredChunk


class BaseRetriever:
    """Default `search_batch` that loops over `search`; subclasses override for speed."""

    name: str = "base"

    def index(self, chunks: Iterable[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: AnalyzedQuery, k: int, version: str = "HEAD") -> list[ScoredChunk]:
        return self.search_batch([query], k, version)[0]

    def search_batch(
        self, queries: Sequence[AnalyzedQuery], k: int, version: str = "HEAD"
    ) -> list[list[ScoredChunk]]:
        return [self.search(q, k, version) for q in queries]
