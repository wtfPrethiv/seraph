import pytest

from fakes import make_chunk
from seraph.retrieval.fusion import fuse, normalize_scores, rrf, weighted
from seraph.types import ScoredChunk

A, B, C = (make_chunk(t) for t in ("alpha", "beta", "gamma"))


def hits(view, pairs):
    return [ScoredChunk(c, s, {view: s}) for c, s in pairs]


RUNS = {
    "lexical": hits("lexical", [(A, 10.0), (B, 5.0)]),
    "semantic": hits("semantic", [(B, 0.9), (C, 0.8)]),
}


def test_rrf_prefers_doc_in_both_views():
    out = rrf(RUNS, k=60)
    assert out[0].chunk == B
    assert out[0].scores == {"lexical": 5.0, "semantic": 0.9}


def test_weighted_respects_weights():
    lex_heavy = weighted(RUNS, {"lexical": 1.0, "semantic": 0.0})
    sem_heavy = weighted(RUNS, {"lexical": 0.0, "semantic": 1.0})
    assert lex_heavy[0].chunk == A
    assert sem_heavy[0].chunk == B


def test_normalize():
    import numpy as np

    assert normalize_scores(np.array([1.0, 3.0]), "minmax").tolist() == [0.0, 1.0]
    assert normalize_scores(np.array([2.0, 2.0]), "zscore").tolist() == [0.0, 0.0]


def test_fuse_dispatch():
    assert fuse({"lexical": RUNS["lexical"]}, "none")[0].chunk == A
    with pytest.raises(ValueError):
        fuse(RUNS, "weighted")
    assert len(fuse(RUNS, "rrf")) == 3
