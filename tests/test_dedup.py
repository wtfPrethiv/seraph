import numpy as np

from fakes import FakeEmbedder, make_chunk, sample_repo
from seraph.config import SeraphConfig
from seraph.retrieval.dedup import evolution_view, lineage_dedup, lineage_history, mmr
from seraph.retrieval.pipeline import Pipeline
from seraph.types import AnalyzedQuery, ScoredChunk


def _parse_hits(store):
    parse = [c for c in store.iter_chunks("*") if c.symbol == "cfg.parse_config"]
    old = next(c for c in parse if c.version_id == "v1")
    new = next(c for c in parse if c.version_id == "HEAD")
    other = next(c for c in store.iter_chunks() if c.symbol == "io.read_file")
    return old, new, other


def test_dedup_keeps_one_per_lineage_preferring_target():
    store, _, versions = sample_repo()
    old, new, other = _parse_hits(store)
    hits = [ScoredChunk(old, 0.9), ScoredChunk(new, 0.8), ScoredChunk(other, 0.5)]
    head = lineage_dedup(hits, versions)
    assert [h.chunk for h in head] == [new, other] and head[0].score == 0.9
    assert lineage_dedup(hits, versions, target_version="v1")[0].chunk == old


def test_dedup_keep_history_orders_newest_first():
    store, _, versions = sample_repo()
    old, new, other = _parse_hits(store)
    hits = [ScoredChunk(old, 0.9), ScoredChunk(other, 0.5), ScoredChunk(new, 0.8)]
    out = lineage_dedup(hits, versions, keep_history=True)
    assert [h.chunk for h in out] == [new, old, other]
    hist = lineage_history(versions, "L_parse")
    assert [x["change_type"] for x in hist] == ["added", "modified"]


def test_evolution_view_matches_commit_messages_and_refs():
    store, _, versions = sample_repo()
    old, new, other = _parse_hits(store)
    cands = [ScoredChunk(new, 1.0), ScoredChunk(other, 0.5)]
    q = AnalyzedQuery.plain("when was strict validation added to config parsing")
    out = evolution_view(q, cands, versions)
    assert [h.chunk for h in out] == [new] and out[0].scores["evolution"] > 0.3


def test_mmr_demotes_near_duplicates():
    a = make_chunk("sort numbers ascending quickly")
    a2 = make_chunk("sort numbers ascending quickly please")
    b = make_chunk("open tcp socket connection")
    hits = [ScoredChunk(a, 1.0), ScoredChunk(a2, 0.99), ScoredChunk(b, 0.9)]
    assert [h.chunk for h in mmr(hits, lam=0.5)] == [a, b, a2]
    emb = FakeEmbedder()
    vec = {c.chunk_hash: v for c, v in zip((a, a2, b), emb.encode_documents([a.text, a2.text, b.text]), strict=True)}
    out = mmr(hits, lam=0.5, vectors=lambda hs: np.stack([vec[h] for h in hs]))
    assert out[1].chunk == b
    assert all(out[i].score > out[i + 1].score for i in range(2))


def test_pipeline_dedup_and_history():
    store, graph, versions = sample_repo()
    cfg = SeraphConfig.model_validate(
        {"retrieval": {"use_dense": True, "fusion": "weighted", "use_dedup": True, "use_evolution": True,
                       "use_query_analyzer": True,
                       "static_weights": {"lexical": 0.4, "semantic": 0.4, "evolution": 0.2}}}
    )
    p = Pipeline.from_config(cfg, store, embedder=FakeEmbedder(), graph=graph, versions=versions)
    resp = p.search("parse config json", top_k=5)
    parse_hits = [r for r in resp.results if r.symbol == "cfg.parse_config"]
    assert len(parse_hits) == 1 and parse_hits[0].version_id == "HEAD"
    evo = p.search("how did parse_config change since v1.0", top_k=5)
    assert evo.query_type == "EVOLUTIONARY"
    assert sum(r.symbol == "cfg.parse_config" for r in evo.results) == 2
    assert any(r.history for r in evo.results)
