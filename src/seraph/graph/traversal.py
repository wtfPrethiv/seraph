"""Graph-expansion retrieval (delta view) and dependency-chain scoring over the CodeGraph protocol."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from seraph.protocols import ChunkStore, CodeGraph
from seraph.types import AnalyzedQuery, Direction, Edge, EdgeKind, QueryType, ScoredChunk, View

EDGE_WEIGHTS: dict[EdgeKind, float] = {
    EdgeKind.CALLS: 1.0,
    EdgeKind.INHERITS: 0.8,
    EdgeKind.DATAFLOW: 0.7,
    EdgeKind.IMPORTS: 0.6,
    EdgeKind.REFERENCES: 0.5,
    EdgeKind.DEFINES: 0.3,
}
ALL_KINDS = set(EDGE_WEIGHTS)

_INCOMING = re.compile(r"\b(who calls|callers?|called by|used by|uses of|who uses|imported by|referenced by|invoked by|subclass(?:es)?)\b", re.I)
_OUTGOING = re.compile(r"\b(callees?|calls?|depends on|dependenc(?:y|ies)|imports?|invokes?|inherits?|uses?)\b", re.I)


def direction_for(query: AnalyzedQuery) -> Direction:
    """Incoming edges for "who calls X", outgoing for "what does X call", both otherwise."""
    text = query.text
    inc = bool(_INCOMING.search(text))
    out = bool(_OUTGOING.search(_INCOMING.sub(" ", text)))
    if inc and not out:
        return Direction.IN
    if out and not inc:
        return Direction.OUT
    return Direction.BOTH


def kinds_for(query: AnalyzedQuery) -> set[EdgeKind]:
    t = query.text.lower()
    if "import" in t:
        return {EdgeKind.IMPORTS}
    if "inherit" in t or "subclass" in t:
        return {EdgeKind.INHERITS}
    if "call" in t or "invok" in t:
        return {EdgeKind.CALLS}
    return ALL_KINDS


def _other(e: Edge, sym: str) -> str:
    return e.dst if e.src == sym else e.src


def expand(
    seeds: Sequence[ScoredChunk],
    graph: CodeGraph,
    store: ChunkStore,
    kinds: set[EdgeKind] | None = None,
    direction: Direction = Direction.BOTH,
    hops: int = 1,
    decay: float = 0.5,
    max_seeds: int = 10,
    include_seeds: bool = True,
) -> list[ScoredChunk]:
    """Spread seed scores over the graph: neighbor = parent * decay^hop * edge_weight (max-aggregated)."""
    kinds = kinds or ALL_KINDS
    top = seeds[:max_seeds]
    if not top:
        return []
    hi = max(s.score for s in top) or 1.0
    best: dict[str, float] = {}
    seed_syms: set[str] = set()
    frontier: list[tuple[str, float]] = []
    for s in top:
        sym = graph.symbol_for_chunk(s.chunk_hash)
        if sym is None:
            continue
        norm = s.score / hi
        seed_syms.add(sym)
        if include_seeds:
            best[sym] = max(best.get(sym, 0.0), norm)
        frontier.append((sym, norm))
    for _ in range(hops):
        nxt: list[tuple[str, float]] = []
        for sym, score in frontier:
            for e in graph.neighbors(sym, kinds, direction):
                other = _other(e, sym)
                val = score * decay * EDGE_WEIGHTS.get(e.kind, 0.5)
                if val > best.get(other, 0.0) and (other not in seed_syms or include_seeds):
                    best[other] = val
                    nxt.append((other, val))
        frontier = nxt

    out = []
    for sym, score in best.items():
        h = graph.chunk_for_symbol(sym)
        if h is None:
            continue
        try:
            chunk = store.get(h)
        except KeyError:
            continue
        out.append(ScoredChunk(chunk, score, {View.GRAPH.value: score}))
    out.sort(key=lambda x: (-x.score, x.chunk_hash))
    return out


@dataclass
class DependencyChain:
    symbols: list[str]
    edges: list[Edge]
    score: float


def dependency_chain_score(path: Sequence[Edge], decay: float = 0.8) -> float:
    s = 1.0
    for i, e in enumerate(path):
        s *= EDGE_WEIGHTS.get(e.kind, 0.5) * (decay if i else 1.0)
    return s


def find_dependencies(
    graph: CodeGraph,
    symbol: str,
    direction: Direction = Direction.OUT,
    kinds: set[EdgeKind] | None = None,
    max_depth: int = 3,
    limit: int = 50,
) -> list[DependencyChain]:
    """All simple chains from `symbol` up to `max_depth`, best-scored first."""
    kinds = kinds or ALL_KINDS
    chains: list[DependencyChain] = []
    q: deque[tuple[list[str], list[Edge]]] = deque([([symbol], [])])
    while q:
        syms, edges = q.popleft()
        if edges:
            chains.append(DependencyChain(syms, edges, dependency_chain_score(edges)))
        if len(edges) >= max_depth:
            continue
        for e in graph.neighbors(syms[-1], kinds, direction):
            nxt = _other(e, syms[-1])
            if nxt not in syms:
                q.append((syms + [nxt], edges + [e]))
    chains.sort(key=lambda c: (-c.score, len(c.edges), c.symbols))
    return chains[:limit]


def wants_graph(query: AnalyzedQuery) -> bool:
    return query.query_type in (QueryType.DEPENDENCY, QueryType.MIXED) or bool(query.dependency_cues)
