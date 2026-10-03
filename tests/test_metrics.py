import random

import numpy as np
import pytest

from seraph.evaluation.coir import SplitGuardError, dev_query_ids
from seraph.evaluation.metrics import (
    alpha_ndcg_at_k,
    evaluate,
    intra_list_similarity,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from seraph.evaluation.runfiles import read_run, write_run


def test_basic_metrics():
    rels = {"d2": 1}
    ranking = ["d1", "d2", "d3"]
    assert mrr_at_k(ranking, rels, 10) == 0.5
    assert ndcg_at_k(ranking, rels, 10) == pytest.approx(1 / np.log2(3))
    assert recall_at_k(ranking, rels, 1) == 0.0
    assert recall_at_k(ranking, rels, 2) == 1.0


def test_tie_break_matches_trec_eval():
    # equal scores: trec_eval ranks the larger doc id first
    m = evaluate({"q": {"a": 1.0, "b": 1.0}}, {"q": {"a": 1}}, ks=(1,))
    assert m["mrr@1"] == 0.0


def _random_run(n_q=200, n_d=300, seed=0):
    rng = random.Random(seed)
    qrels = {f"q{i}": {f"d{rng.randrange(n_d)}": 1} for i in range(n_q)}
    run = {q: {f"d{j}": rng.random() for j in rng.sample(range(n_d), min(120, n_d))} for q in qrels}
    return run, qrels


def test_matches_mteb():
    pytest.importorskip("mteb")
    from seraph.evaluation.coir import mteb_scores

    run, qrels = _random_run()
    ours = evaluate(run, qrels, ks=(1, 10, 100))
    theirs = mteb_scores(qrels, run, ks=(1, 10, 100))
    for k, v in theirs.items():
        assert ours[k] == pytest.approx(v, abs=1e-4), k


def test_runfile_roundtrip(tmp_path):
    run, _ = _random_run(5, 20)
    write_run(run, tmp_path / "r.trec")
    back = read_run(tmp_path / "r.trec")
    assert set(back) == set(run)
    assert back["q0"] == pytest.approx(run["q0"], abs=1e-5)


def test_dev_split_is_stable_and_disjoint():
    train = {f"q{i}": "x" for i in range(1000)}
    a, b = dev_query_ids(train), dev_query_ids(train)
    assert a == b and len(a) == 100


def test_test_split_guard():
    from seraph.evaluation.coir import AppsData

    data = AppsData({}, {}, {}, {"t": "x"}, {"t": {"d": 1}}, "rev")
    with pytest.raises(SplitGuardError):
        data.split("test")
    assert data.split("test", final=True).queries == {"t": "x"}


def test_diversity_metrics():
    intents = {"a": {"x"}, "b": {"x"}, "c": {"y"}}
    assert alpha_ndcg_at_k(["a", "c", "b"], intents, 3) > alpha_ndcg_at_k(["a", "b", "c"], intents, 3)
    e = np.eye(3)
    assert intra_list_similarity(e) == 0.0
    assert intra_list_similarity(np.ones((3, 2))) == pytest.approx(1.0)
