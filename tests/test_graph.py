import pytest

from fakes import sample_repo
from seraph.graph.traversal import (
    dependency_chain_score,
    direction_for,
    expand,
    find_dependencies,
    kinds_for,
)
from seraph.types import AnalyzedQuery, Direction, Edge, EdgeKind, ScoredChunk


def seed(store, symbol, score=1.0):
    c = next(c for c in store.iter_chunks() if c.symbol == symbol)
    return ScoredChunk(c, score, {"semantic": score})


@pytest.mark.parametrize(
    ("q", "d"),
    [
        ("who calls parse_config", Direction.IN),
        ("callers of read_file", Direction.IN),
        ("what does load call", Direction.OUT),
        ("dependencies of parse_config", Direction.OUT),
        ("config parsing", Direction.BOTH),
    ],
)
def test_direction(q, d):
    assert direction_for(AnalyzedQuery.plain(q)) == d


def test_kinds():
    assert kinds_for(AnalyzedQuery.plain("who imports json")) == {EdgeKind.IMPORTS}
    assert kinds_for(AnalyzedQuery.plain("who calls x")) == {EdgeKind.CALLS}


def test_expand_callers_and_callees_with_decay():
    store, graph, _ = sample_repo()
    s = [seed(store, "cfg.parse_config")]
    both = {h.chunk.symbol: h.score for h in expand(s, graph, store, hops=1, decay=0.5)}
    assert both == {"cfg.parse_config": 1.0, "io.read_file": 0.5, "app.load": 0.5}
    callers = expand(s, graph, store, direction=Direction.IN, include_seeds=False)
    assert [h.chunk.symbol for h in callers] == ["app.load"]
    two_hop = {h.chunk.symbol: h.score for h in expand([seed(store, "app.load")], graph, store, direction=Direction.OUT, hops=2)}
    assert two_hop["io.read_file"] == pytest.approx(0.25)
    assert "graph" in callers[0].scores


def test_dependency_chains():
    _, graph, _ = sample_repo()
    chains = find_dependencies(graph, "app.load", Direction.OUT)
    assert [c.symbols for c in chains] == [["app.load", "cfg.parse_config"], ["app.load", "cfg.parse_config", "io.read_file"]]
    assert chains[0].score > chains[1].score
    assert dependency_chain_score([Edge("a", "b", EdgeKind.IMPORTS)]) == pytest.approx(0.6)
