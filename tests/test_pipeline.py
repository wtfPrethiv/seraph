from fakes import FakeEmbedder, sample_repo
from seraph.config import SeraphConfig
from seraph.retrieval.pipeline import Pipeline
from seraph.types import AnalyzedQuery


def make_pipeline(**retrieval):
    store, _, _ = sample_repo()
    cfg = SeraphConfig.model_validate({"retrieval": {"use_dense": True, **retrieval}})
    return Pipeline.from_config(cfg, store, embedder=FakeEmbedder())


def test_hybrid_rrf_has_both_view_scores():
    p = make_pipeline(fusion="rrf")
    hits = p.search_batch([AnalyzedQuery.plain("parse config strict")], 3)[0]
    assert hits[0].chunk.symbol == "cfg.parse_config"
    assert {"lexical", "semantic"} <= set(hits[0].scores)


def test_weighted_fusion_runs():
    p = make_pipeline(fusion="weighted", static_weights={"lexical": 0.5, "semantic": 0.5})
    assert p.search_batch([AnalyzedQuery.plain("read file")], 2)[0][0].chunk.symbol == "io.read_file"


def test_public_search_response_shape():
    import json

    import pytest

    from seraph.retrieval.pipeline import register_repo, search

    p = make_pipeline(fusion="weighted", static_weights={"lexical": 0.4, "semantic": 0.6})
    register_repo("sample", p)
    resp = search("parse config strict validate", repo="sample", top_k=2)
    d = resp.to_dict()
    json.dumps(d)
    assert d["results"][0]["symbol"] == "cfg.parse_config"
    assert set(d["results"][0]["retrieval_scores"]) >= {"lexical", "semantic"}
    assert d["weights"] == {"lexical": 0.4, "semantic": 0.6}
    with pytest.raises(KeyError, match="not indexed"):
        search("x", repo="missing")


def test_analyzer_and_decomposition_in_pipeline():
    p = make_pipeline(
        fusion="weighted",
        static_weights={"lexical": 0.5, "semantic": 0.5},
        use_query_analyzer=True,
        use_decomposition=True,
    )
    q = "read the file at path; then parse config with strict validation"
    resp = p.search(q, top_k=3)
    assert resp.query_type == "SEMANTIC"
    assert len(resp.sub_queries) == 3  # full + two clauses
    assert {"cfg.parse_config", "io.read_file"} <= {r.symbol for r in resp.results}
    batch = p.search_batch([AnalyzedQuery.plain(q, qid="x")], 3)[0]
    assert [h.chunk.symbol for h in batch] == [r.symbol for r in resp.results]


def test_graph_expansion_surfaces_callers():
    store, graph, _ = sample_repo()
    cfg = SeraphConfig.model_validate(
        {"retrieval": {"use_dense": False, "fusion": "weighted", "use_graph_expansion": True,
                       "static_weights": {"lexical": 0.5, "graph": 0.5}}}
    )
    p = Pipeline.from_config(cfg, store, graph=graph)
    hits = p.search_batch([AnalyzedQuery.plain("who calls read_file")], 5)[0]
    syms = [h.chunk.symbol for h in hits]
    assert "cfg.parse_config" in syms[:2]
    assert any("graph" in h.scores for h in hits)


class FlipReranker:
    """Reverses the candidate order; checks the rerank stage is applied and the tail kept."""

    name = "flip"

    def rerank_batch(self, queries, candidates, top_k):
        out = []
        for cands in candidates:
            n = len(cands)
            out.append([type(c)(c.chunk, float(i), {**c.scores, "rerank": float(i)}) for i, c in enumerate(cands)][::-1][:top_k])
            assert n
        return out


def test_rerank_stage_and_tail():
    store, _, _ = sample_repo()
    cfg = SeraphConfig.model_validate(
        {"retrieval": {"use_dense": True, "fusion": "rrf", "use_reranker": True, "rerank_depth": 2}}
    )
    p = Pipeline.from_config(cfg, store, embedder=FakeEmbedder(), reranker=FlipReranker())
    base = make_pipeline(fusion="rrf").search_batch([AnalyzedQuery.plain("parse config")], 4)[0]
    hits = p.search_batch([AnalyzedQuery.plain("parse config")], 4)[0]
    assert hits[0].chunk == base[1].chunk and hits[1].chunk == base[0].chunk
    assert [h.chunk for h in hits[2:]] == [h.chunk for h in base[2:]]
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
