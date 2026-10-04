import numpy as np

from fakes import FakeEmbedder, make_chunk
from seraph.config import SeraphConfig
from seraph.memory import InMemoryChunkStore
from seraph.query.weights import (
    LearnedWeights,
    RuleWeights,
    cell_utilities,
    query_features,
    simplex_grid,
    view_matrix,
)
from seraph.retrieval.fusion import weighted
from seraph.retrieval.pipeline import Pipeline
from seraph.types import AnalyzedQuery, ScoredChunk


def _hits(view, pairs):
    return [ScoredChunk(make_chunk(t), s, {view: s}) for t, s in pairs]


def test_simplex_grid():
    g = simplex_grid(["a", "b", "c"], 0.1)
    assert len(g) == 66 and np.allclose(g.sum(axis=1), 1.0)


def test_rule_weights_follow_identifiers():
    rw = RuleWeights()
    ident = rw.weights(AnalyzedQuery.plain("parse_config load_yaml"), ["lexical", "semantic"])
    prose = rw.weights(AnalyzedQuery.plain("find the shortest path between two cities in a weighted road network"),
                       ["lexical", "semantic"])
    assert abs(sum(ident.values()) - 1) < 1e-9
    assert ident["lexical"] > prose["lexical"] and prose["semantic"] > ident["semantic"]


def test_view_matrix_and_utilities_match_weighted_fusion():
    lex = _hits("lexical", [("a", 5.0), ("b", 3.0), ("c", 1.0)])
    sem = _hits("semantic", [("c", 0.9), ("d", 0.8), ("a", 0.1)])
    runs = {"lexical": lex, "semantic": sem}
    views = ["lexical", "semantic"]
    hashes, m = view_matrix(runs, views)
    fused = {h.chunk_hash: h.score for h in weighted(runs, {"lexical": 0.3, "semantic": 0.7})}
    for h, row in zip(hashes, m, strict=True):
        assert abs(row @ [0.3, 0.7] - fused[h]) < 1e-9
    gold = {sem[1].chunk_hash: 1}
    u = cell_utilities([runs], [gold], views, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert u[0, 1] > u[0, 0]


def test_learned_weights_pick_per_query_cell():
    views = ["lexical", "semantic"]
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    model = LearnedWeights(views, step=0.5, kind="ridge")
    util = np.zeros((200, len(model.grid)))
    lex_cell = int(np.argmax(model.grid[:, 0]))
    sem_cell = int(np.argmax(model.grid[:, 1]))
    util[x[:, 0] > 0, lex_cell] = 1.0
    util[x[:, 0] <= 0, sem_cell] = 1.0
    model.fit(x, util)
    pred = model.predict_cells(np.array([[2.0, 0, 0], [-2.0, 0, 0]]))
    assert list(pred) == [lex_cell, sem_cell]
    top = LearnedWeights(views, step=0.5, kind="ridge", top_cells=1).fit(x, util)
    assert len(set(top.predict_cells(x))) == 1


def test_query_features_shape_is_stable():
    runs = {"lexical": _hits("lexical", [("a", 2.0)]), "semantic": []}
    f1 = query_features(AnalyzedQuery.plain("short"), runs, ["lexical", "semantic"])
    f2 = query_features(AnalyzedQuery.plain("a much longer query about graphs and trees"), runs, ["lexical", "semantic"])
    assert f1.shape == f2.shape and np.isfinite(f1).all()


def test_pipeline_adaptive_rules():
    chunks = [make_chunk("def parse_config(path): return json.load(open(path))"),
              make_chunk("def add(a, b): return a + b")]
    cfg = SeraphConfig.model_validate({"retrieval": {"use_dense": True, "fusion": "adaptive", "adaptive_weights": "rules"}})
    p = Pipeline.from_config(cfg, InMemoryChunkStore(chunks), embedder=FakeEmbedder())
    (q,) = [AnalyzedQuery.plain("parse_config")]
    hits = p.search_batch([q], 2)[0]
    assert hits[0].chunk == chunks[0]
    assert set(q.weights) == {"lexical", "semantic"} and abs(sum(q.weights.values()) - 1) < 1e-9
