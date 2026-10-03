from fakes import FakeEmbedder, FakeRetriever, sample_repo
from seraph.config import EMBEDDERS, RERANKERS, SeraphConfig, load_config
from seraph.protocols import ChunkStore, CodeGraph, Embedder, Retriever, VersionStore
from seraph.types import AnalyzedQuery, ChangeType, Direction, EdgeKind


def test_fakes_satisfy_protocols():
    store, graph, versions = sample_repo()
    assert isinstance(store, ChunkStore)
    assert isinstance(graph, CodeGraph)
    assert isinstance(versions, VersionStore)
    assert isinstance(FakeRetriever(), Retriever)
    assert isinstance(FakeEmbedder(), Embedder)


def test_graph_neighbors_and_resolve():
    _, graph, _ = sample_repo()
    callees = graph.neighbors("cfg.parse_config", {EdgeKind.CALLS}, Direction.OUT)
    callers = graph.neighbors("cfg.parse_config", {EdgeKind.CALLS}, "in")
    assert [e.dst for e in callees] == ["io.read_file"]
    assert [e.src for e in callers] == ["app.load"]
    assert graph.resolve("parse_config") == ["cfg.parse_config"]


def test_lineage_and_diff():
    _, _, versions = sample_repo()
    nodes = versions.lineage("L_parse")
    assert [n.version_id for n in nodes] == ["v1", "v2"]
    d = versions.diff_symbol("cfg.parse_config", "v1", "v2")
    assert d.change_type == ChangeType.MODIFIED and "strict" in d.unified_diff


def test_changed_since():
    store, _, _ = sample_repo()
    cs = store.changed_since("v1")
    assert len(cs.removed) == 1 and len(cs.added) == 4


def test_fake_retriever_search_batch():
    store, _, _ = sample_repo()
    r = FakeRetriever()
    r.index(store.iter_chunks())
    out = r.search_batch([AnalyzedQuery.plain("parse config"), AnalyzedQuery.plain("zzz")], 3)
    assert out[0] and not out[1]


def test_config_extends(tmp_path):
    (tmp_path / "base.yaml").write_text("retrieval:\n  use_dense: true\n  rrf_k: 10\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nname: c\nretrieval:\n  rrf_k: 20\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.name == "c" and cfg.retrieval.use_dense and cfg.retrieval.rrf_k == 20
    assert isinstance(SeraphConfig().retrieval.static_weights, dict)
    assert "qwen3-emb-0.6b" in EMBEDDERS and "bge-reranker-v2-m3" in RERANKERS
