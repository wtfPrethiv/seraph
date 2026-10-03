"""In-memory implementations of the storage protocols.

Used for snippet benchmarks (AppsRetrieval has no repo) and as the base for test fakes.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from collections.abc import Iterable, Iterator

from seraph.types import (
    ChangeSet,
    ChangeType,
    Chunk,
    Direction,
    Edge,
    EdgeKind,
    LineageNode,
    SymbolDiff,
    Version,
)


class InMemoryChunkStore:
    def __init__(self, chunks: Iterable[Chunk] = ()) -> None:
        self._by_hash: dict[str, Chunk] = {}
        self._by_version: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            self.add(c)

    def add(self, chunk: Chunk) -> None:
        self._by_hash[chunk.chunk_hash] = chunk
        self._by_version[chunk.version_id].append(chunk.chunk_hash)

    def iter_chunks(self, version: str = "HEAD") -> Iterator[Chunk]:
        if version == "*":
            yield from self._by_hash.values()
            return
        for h in self._by_version.get(version, []):
            yield self._by_hash[h]

    def get(self, chunk_hash: str) -> Chunk:
        return self._by_hash[chunk_hash]

    def changed_since(self, version: str) -> ChangeSet:
        old = set(self._by_version.get(version, []))
        new = set(self._by_version.get("HEAD", []))
        return ChangeSet(
            added=sorted(new - old), removed=sorted(old - new), unchanged=sorted(new & old)
        )

    def __len__(self) -> int:
        return len(self._by_hash)


class InMemoryCodeGraph:
    def __init__(self) -> None:
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)
        self._sym_by_chunk: dict[str, str] = {}
        self._chunk_by_sym: dict[str, str] = {}

    def add_symbol(self, symbol_id: str, chunk_hash: str) -> None:
        self._sym_by_chunk[chunk_hash] = symbol_id
        self._chunk_by_sym[symbol_id] = chunk_hash

    def add_edge(self, src: str, dst: str, kind: EdgeKind) -> None:
        e = Edge(src, dst, kind)
        self._out[src].append(e)
        self._in[dst].append(e)

    def neighbors(
        self, symbol_id: str, kinds: set[EdgeKind], direction: Direction | str
    ) -> list[Edge]:
        direction = Direction(direction)
        edges: list[Edge] = []
        if direction in (Direction.OUT, Direction.BOTH):
            edges += self._out.get(symbol_id, [])
        if direction in (Direction.IN, Direction.BOTH):
            edges += self._in.get(symbol_id, [])
        return [e for e in edges if e.kind in kinds]

    def symbol_for_chunk(self, chunk_hash: str) -> str | None:
        return self._sym_by_chunk.get(chunk_hash)

    def chunk_for_symbol(self, symbol_id: str) -> str | None:
        return self._chunk_by_sym.get(symbol_id)

    def resolve(self, name: str) -> list[str]:
        return [s for s in self._chunk_by_sym if s == name or s.rsplit(".", 1)[-1] == name]


class InMemoryVersionStore:
    def __init__(self, versions: Iterable[Version] = ()) -> None:
        self._versions = list(versions)
        self._lineages: dict[str, list[LineageNode]] = defaultdict(list)
        self._texts: dict[str, str] = {}

    def add_version(self, v: Version) -> None:
        self._versions.append(v)

    def add_lineage_node(self, node: LineageNode, text: str | None = None) -> None:
        self._lineages[node.lineage_id].append(node)
        if node.chunk_hash and text is not None:
            self._texts[node.chunk_hash] = text

    def versions(self) -> list[Version]:
        return sorted(self._versions, key=lambda v: v.timestamp)

    def version_rank(self) -> dict[str, int]:
        return {v.id: i for i, v in enumerate(self.versions())}

    def lineage(self, lineage_id: str) -> list[LineageNode]:
        rank = self.version_rank()
        return sorted(self._lineages.get(lineage_id, []), key=lambda n: rank.get(n.version_id, 0))

    def diff_symbol(self, symbol: str, a: str, b: str) -> SymbolDiff:
        nodes = [n for ns in self._lineages.values() for n in ns if n.symbol_id == symbol]
        na = next((n for n in nodes if n.version_id == a), None)
        nb = next((n for n in nodes if n.version_id == b), None)
        ta = self._texts.get(na.chunk_hash) if na and na.chunk_hash else None
        tb = self._texts.get(nb.chunk_hash) if nb and nb.chunk_hash else None
        if ta is None and tb is not None:
            ct = ChangeType.ADDED
        elif tb is None and ta is not None:
            ct = ChangeType.DELETED
        elif ta == tb:
            ct = ChangeType.UNCHANGED
        else:
            ct = ChangeType.MODIFIED
        diff = "".join(
            difflib.unified_diff((ta or "").splitlines(True), (tb or "").splitlines(True), a, b)
        )
        return SymbolDiff(symbol, a, b, ct, ta, tb, diff)
