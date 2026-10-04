"""Rank chunks from a `VersionedIndex` with the retrieval `Pipeline` (used by the CLI and MCP server)."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from typing import Any

from seraph import index as vindex
from seraph.config import SeraphConfig, load_config
from seraph.types import ChangeSet, Chunk

HISTORY = "*"


def to_chunk(c: vindex.Chunk) -> Chunk:
    """Ranking text is prefixed with path and symbol; results show the original `vindex.Chunk.text`."""
    return Chunk(
        chunk_hash=c.occurrence_id,
        text=f"{c.path} {c.symbol or ''}\n{c.text}",
        lang=c.language,
        file=c.path,
        symbol=c.symbol,
        kind=c.kind,
        start_line=c.start_line,
        end_line=c.end_line,
        version_id=c.commit,
        # Until the index tracks lineage, a symbol's identity across commits is its path + name.
        lineage_id=f"{c.path}::{c.symbol}" if c.symbol else None,
    )


class IndexChunkStore:
    """`ChunkStore` over a snapshot of index chunks; `head` is the commit that "HEAD" refers to."""

    def __init__(self, chunks: Iterable[vindex.Chunk], head: str) -> None:
        self.head = head
        self._source: dict[str, vindex.Chunk] = {}
        self._chunks: dict[str, Chunk] = {}
        self._by_commit: dict[str, list[str]] = {}
        for c in chunks:
            self._source[c.occurrence_id] = c
            self._chunks[c.occurrence_id] = to_chunk(c)
            self._by_commit.setdefault(c.commit, []).append(c.occurrence_id)

    @classmethod
    def from_index(cls, index: vindex.VersionedIndex, ref: str = "HEAD", include_history: bool = False) -> IndexChunkStore:
        return cls(index.iter_chunks(ref, include_history), index.resolve(ref))

    def _commit(self, version: str) -> str:
        if version == "HEAD":
            return self.head
        matches = [c for c in self._by_commit if c.startswith(version)]
        if len(matches) != 1:
            raise KeyError(f"version {version!r} is not in this snapshot")
        return matches[0]

    def iter_chunks(self, version: str = "HEAD") -> Iterator[Chunk]:
        if version == HISTORY:
            yield from self._chunks.values()
            return
        for oid in self._by_commit.get(self._commit(version), []):
            yield self._chunks[oid]

    def get(self, chunk_hash: str) -> Chunk:
        return self._chunks[chunk_hash]

    def source(self, chunk_hash: str) -> vindex.Chunk:
        return self._source[chunk_hash]

    def changed_since(self, version: str) -> ChangeSet:
        old = {self._source[o].content_hash for o in self._by_commit.get(self._commit(version), [])}
        head = self._by_commit.get(self.head, [])
        new = {self._source[o].content_hash for o in head}
        return ChangeSet(
            added=sorted(o for o in head if self._source[o].content_hash not in old),
            removed=sorted(old - new),
            unchanged=sorted(o for o in head if self._source[o].content_hash in old),
        )

    def __len__(self) -> int:
        return len(self._chunks)


def service_config() -> SeraphConfig:
    """`SERAPH_CONFIG` (YAML) if set; otherwise BM25 with the code-aware tokenizer, no API calls."""
    path = os.environ.get("SERAPH_CONFIG")
    return load_config(path) if path else SeraphConfig()


_PIPELINES: dict[tuple, tuple[Any, IndexChunkStore]] = {}


def _pipeline(index: vindex.VersionedIndex, commit: str, include_history: bool, cfg: SeraphConfig):
    from seraph.retrieval.pipeline import Pipeline

    scope = tuple(index.indexed_versions()) if include_history else (commit,)
    key = (str(index.repo), scope, include_history, hashlib.sha1(cfg.model_dump_json().encode()).hexdigest())
    if key not in _PIPELINES:
        store = IndexChunkStore.from_index(index, commit, include_history)
        pipe = Pipeline.from_config(cfg, store, HISTORY if include_history else "HEAD") if len(store) else None
        _PIPELINES[key] = (pipe, store)
    return _PIPELINES[key]


def search_index(
    index: vindex.VersionedIndex,
    query: str,
    ref: str = "HEAD",
    limit: int = 10,
    include_history: bool = False,
    cfg: SeraphConfig | None = None,
) -> dict[str, Any]:
    """Search an indexed version (indexing it first if needed) and return JSON-ready results."""
    t0 = time.perf_counter()
    stats = index.index_commit(ref)
    cfg = cfg or service_config()
    pipe, store = _pipeline(index, stats.commit, include_history, cfg)
    t1 = time.perf_counter()
    out: dict[str, Any] = {
        "query": query,
        "requested_version": ref,
        "resolved_commit": stats.commit,
        "index": {**asdict(stats), "ms": round((t1 - t0) * 1000, 1)},
        "results": [],
    }
    if pipe is None or not query.strip() or limit < 1:
        return out
    resp = pipe.search(query, top_k=limit * 4 if include_history else limit, include_history=include_history)
    results: list[dict[str, Any]] = []
    by_content: dict[str, dict[str, Any]] = {}
    for r in resp.results:
        source = store.source(r.chunk_hash)
        seen = by_content.get(source.content_hash) if include_history else None
        if seen is not None:
            seen["versions"].append(source.commit)
            continue
        hit = {"score": round(r.score, 4), **asdict(source), "retrieval_scores": r.retrieval_scores}
        if include_history:
            hit["versions"] = [source.commit]
            by_content[source.content_hash] = hit
        if len(results) < limit:
            results.append(hit)
    out["search_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    out["query_type"] = resp.query_type
    out["results"] = results
    return out
