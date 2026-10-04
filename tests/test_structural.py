from fakes import FakeEmbedder, make_chunk
from seraph.config import SeraphConfig
from seraph.memory import InMemoryChunkStore
from seraph.retrieval.pipeline import Pipeline
from seraph.retrieval.structural import StructuralModel, StructuralScorer, extract_traits
from seraph.types import AnalyzedQuery, ScoredChunk

HEAP = "import heapq\nh=[]\nfor x in map(int, input().split()):\n    heapq.heappush(h, x)\nprint(heapq.heappop(h))"
REC = "def f(n):\n    return 1 if n < 2 else f(n-1) + f(n-2)\nprint(f(int(input())))"
GRID = "n=int(input())\ng=[input() for _ in range(n)]\nfor i in range(n):\n    for j in range(n):\n        print(g[i][j])"


def test_extract_traits():
    assert {"imp:heapq", "heapq", "io_input", "loop", "map_int", "print"} <= extract_traits(HEAP)
    assert "recursion" in extract_traits(REC) and "loop" not in extract_traits(REC)
    assert {"loop_nested2", "subscript_2d", "io_grid"} <= extract_traits(GRID)
    assert "loop_nested2" not in extract_traits(HEAP)
    assert extract_traits("def broken(:\n  pass") is not None


def _model():
    qs = [f"smallest element priority queue {i}" for i in range(6)]
    qs += [f"fibonacci recursive sequence {i}" for i in range(6)]
    qs += [f"grid of characters rows columns {i}" for i in range(6)]
    docs = [HEAP] * 6 + [REC] * 6 + [GRID] * 6
    return StructuralModel().fit(qs, docs)


def test_scorer_prefers_matching_structure():
    model = _model()
    chunks = [make_chunk(HEAP), make_chunk(REC), make_chunk(GRID)]
    scorer = StructuralScorer(model)
    scorer.index(chunks)
    cands = [ScoredChunk(c, 1.0) for c in chunks]
    queries = [AnalyzedQuery.plain(t) for t in ("priority queue smallest", "fibonacci recursive", "grid rows")]
    out = scorer.rescore(queries, [cands] * 3)
    assert [o[0].chunk for o in out] == chunks
    assert all(o[0].scores["structural"] > o[1].scores["structural"] for o in out)


def test_pipeline_structural_view(tmp_path):
    path = tmp_path / "s.pkl"
    _model().save(path)
    chunks = [make_chunk(HEAP), make_chunk(REC), make_chunk(GRID)]
    cfg = SeraphConfig.model_validate(
        {"retrieval": {"use_dense": True, "fusion": "weighted", "use_structural": True,
                       "structural_model_path": str(path),
                       "static_weights": {"lexical": 0.0, "semantic": 0.0, "structural": 1.0}}}
    )
    p = Pipeline.from_config(cfg, InMemoryChunkStore(chunks), embedder=FakeEmbedder())
    hits = p.search("fibonacci recursive", top_k=3).results
    assert hits[0].chunk_hash == chunks[1].chunk_hash and "structural" in hits[0].retrieval_scores
