from types import SimpleNamespace

import numpy as np
import pytest

from fakes import FakeEmbedder, make_chunk

pytest.importorskip("google.genai")

from seraph.backends.common import MissingAPIKeyError, RateLimiter, with_retries  # noqa: E402
from seraph.backends.gemini import GeminiEmbedder, GeminiListwiseReranker, parse_order  # noqa: E402
from seraph.config import EMBEDDERS, RERANKERS  # noqa: E402
from seraph.types import ScoredChunk  # noqa: E402


class FakeModels:
    def __init__(self, ranking: str = "[2, 1]") -> None:
        self.embed_calls: list[int] = []
        self.gen_calls = 0
        self.ranking = ranking
        self._emb = FakeEmbedder(dim=768)

    def embed_content(self, model, contents, config):
        self.embed_calls.append(len(contents))
        vecs = self._emb.encode_documents(contents)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v.tolist()) for v in vecs])

    def generate_content(self, model, contents, config):
        self.gen_calls += 1
        return SimpleNamespace(text=self.ranking)


def fake_client(**kw):
    return SimpleNamespace(models=FakeModels(**kw))


def spec(name, registry, **opts):
    s = registry[name].model_copy(deep=True)
    s.options.update({"rpm": 0, **opts})
    return s


def test_embedder_caches_each_text(tmp_path):
    client = fake_client()
    e = GeminiEmbedder(spec("gemini-embedding-001", EMBEDDERS, batch=2), tmp_path, client=client)
    a = e.encode_documents(["sort numbers", "open socket", "read file"])
    assert a.shape == (3, 768) and np.allclose(np.linalg.norm(a, axis=1), 1)
    assert client.models.embed_calls == [2, 1]
    # new instance, same cache: only the unseen text is sent
    e2 = GeminiEmbedder(spec("gemini-embedding-001", EMBEDDERS, batch=2), tmp_path, client=client)
    b = e2.encode_documents(["read file", "sort numbers", "parse json"])
    assert client.models.embed_calls == [2, 1, 1]
    assert np.allclose(b[0], a[2])


def test_listwise_reranker_orders_and_caches(tmp_path):
    client = fake_client(ranking="[2, 1]")
    r = GeminiListwiseReranker(spec("gemini-flash-lite-listwise", RERANKERS, window=2), tmp_path, client=client)
    a, b, c = (make_chunk(t) for t in ("aaa", "bbb", "ccc"))
    cands = [ScoredChunk(a, 0.9, {}), ScoredChunk(b, 0.8, {}), ScoredChunk(c, 0.1, {})]
    out = r.rerank("q", cands, 3)
    assert [h.chunk for h in out] == [b, a, c]
    assert out[0].scores["rerank"] > out[1].scores["rerank"] and out[2].score < out[1].score
    r.rerank("q", cands, 3)
    assert client.models.gen_calls == 1


def test_parse_order_is_robust():
    assert parse_order("[3, 1, 3, 9]", 3) == [2, 0, 1]
    assert parse_order('{"ranking": [2]}', 2) == [1, 0]
    assert parse_order("best is 2 then 1", 2) == [1, 0]
    assert parse_order("", 2) == [0, 1]


def test_missing_key_message(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    e = GeminiEmbedder(spec("gemini-embedding-001", EMBEDDERS), tmp_path)
    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
        e.encode_queries(["x"])


def test_fallback_only_without_key(tmp_path, monkeypatch):
    from seraph.retrieval.semantic import LazyLocalEmbedder, load_embedder

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fb = load_embedder("gemini-embedding-001", cache_dir=tmp_path, fallback="qwen3-emb-0.6b")
    assert isinstance(fb, LazyLocalEmbedder) and fb._model is None  # nothing loaded yet
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(load_embedder("gemini-embedding-001", cache_dir=tmp_path, fallback="qwen3-emb-0.6b"), GeminiEmbedder)


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    assert with_retries(flaky) == "ok" and calls["n"] == 3
    with pytest.raises(ValueError):
        with_retries(lambda: (_ for _ in ()).throw(ValueError("bad request")))
    RateLimiter(0).wait()


def test_daily_quota_stops_immediately(monkeypatch):
    from seraph.backends.common import QuotaExhaustedError

    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def exhausted():
        calls["n"] += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(QuotaExhaustedError):
        with_retries(exhausted)
    assert calls["n"] == 1
