import numpy as np

from fakes import FakeEmbedder
from seraph.evaluation.ablations import Ablation, Rung, render
from seraph.retrieval.semantic import CachedEmbedder


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode_queries(self, texts):
        self.calls += 1
        return super().encode_queries(texts)


def test_cached_embedder_assembles_subsets(tmp_path):
    inner = CountingEmbedder()
    emb = CachedEmbedder(inner, tmp_path)
    texts = [f"query number {i}" for i in range(20)]
    full = emb.encode_queries(texts)
    assert inner.calls == 1
    fresh = CachedEmbedder(inner, tmp_path)
    sub = fresh.encode_queries(texts[3:13])
    few = fresh.encode_queries(texts[:2])
    assert inner.calls == 1
    assert np.allclose(sub, full[3:13]) and np.allclose(few, full[:2])
    fresh.encode_queries(texts[:9] + ["never seen before"])
    assert inner.calls == 2


def test_ablation_load_and_render(tmp_path):
    spec = tmp_path / "a.yaml"
    spec.write_text(
        "splits: [dev]\nladder:\n"
        "  - {label: A, config: a.yaml}\n"
        "  - {label: B, config: b.yaml}\n"
        "  - {label: C, config: c.yaml, base: A, api: true}\n"
        "not_applicable:\n  - {label: D, reason: no graph}\n"
    )
    ab = Ablation.load(spec)
    assert [r.base for r in ab.rungs] == [None, "A", "A"] and ab.rungs[2].api
    def res(x):
        return {"metrics": {"ndcg@10": x, "recall@100": 0.9, "latency_p50_ms": 1.0}}

    md = render(ab, {("A", "dev"): res(0.5), ("B", "dev"): res(0.6), ("C", "dev"): "skipped (quota)"})
    assert "0.6000 (+0.1000)" in md and "skipped (quota)" in md and "no graph" in md
    assert isinstance(ab.rungs[0], Rung)
