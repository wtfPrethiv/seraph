"""Shared data types. Changing anything here needs review from both owners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class EdgeKind(StrEnum):
    CALLS = "calls"
    IMPORTS = "imports"
    DEFINES = "defines"
    REFERENCES = "references"
    INHERITS = "inherits"
    DATAFLOW = "dataflow"


class Direction(StrEnum):
    OUT = "out"
    IN = "in"
    BOTH = "both"


class QueryType(StrEnum):
    LEXICAL = "LEXICAL"
    SEMANTIC = "SEMANTIC"
    STRUCTURAL = "STRUCTURAL"
    DEPENDENCY = "DEPENDENCY"
    EVOLUTIONARY = "EVOLUTIONARY"
    MIXED = "MIXED"


class ChangeType(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    RENAMED = "renamed"
    MOVED = "moved"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class View(StrEnum):
    """Retrieval views; their fusion weights are alpha..epsilon in that order."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    GRAPH = "graph"
    EVOLUTION = "evolution"


VIEWS: tuple[View, ...] = tuple(View)


@dataclass(frozen=True)
class Chunk:
    chunk_hash: str
    text: str
    lang: str
    file: str
    symbol: str | None
    kind: str  # function/class/method/file_header/snippet
    start_line: int
    end_line: int
    version_id: str = "HEAD"
    lineage_id: str | None = None


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def chunk_hash(self) -> str:
        return self.chunk.chunk_hash


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: EdgeKind


@dataclass(frozen=True)
class Version:
    id: str
    commit: str
    tag: str | None = None
    parent: str | None = None
    timestamp: int = 0


@dataclass(frozen=True)
class LineageNode:
    lineage_id: str
    symbol_id: str
    version_id: str
    chunk_hash: str | None
    change_type: ChangeType
    commit_message: str = ""


@dataclass
class ChangeSet:
    """Chunks that differ between two versions, keyed by content hash."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


@dataclass
class SymbolDiff:
    symbol: str
    version_a: str
    version_b: str
    change_type: ChangeType
    old_text: str | None = None
    new_text: str | None = None
    unified_diff: str = ""


@dataclass
class SubQuery:
    text: str
    facet: str = "core"  # core/input/output/constraints/identifier
    weight: float = 1.0


@dataclass
class AnalyzedQuery:
    raw: str
    text: str
    query_type: QueryType = QueryType.SEMANTIC
    identifiers: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    version_refs: list[str] = field(default_factory=list)
    dependency_cues: list[str] = field(default_factory=list)
    sub_queries: list[SubQuery] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    type_probs: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] | None = None
    qid: str | None = None

    @classmethod
    def plain(cls, text: str, qid: str | None = None) -> AnalyzedQuery:
        return cls(raw=text, text=text, qid=qid)


@dataclass
class SearchResult:
    chunk_hash: str
    file: str
    symbol: str | None
    kind: str
    start_line: int
    end_line: int
    version_id: str
    score: float
    retrieval_scores: dict[str, float]
    snippet: str
    lineage_id: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SearchResponse:
    query: str
    query_type: str
    version: str
    results: list[SearchResult]
    sub_queries: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
