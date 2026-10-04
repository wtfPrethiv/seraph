import numpy as np
import pytest

from fakes import FakeEmbedder, make_chunk
from seraph.retrieval.semantic import CachedEmbedder, DenseRetriever
from seraph.retrieval.vectors import VectorIndex
from seraph.types import AnalyzedQuery


def test_vector_index_numpy_matches_faiss():
    rng = np.random.default_rng(0)
    docs, qs = rng.normal(size=(50, 8)), rng.normal(size=(3, 8))
    a, b = VectorIndex(8, use_faiss=False), VectorIndex(8)
    a.add(docs)
    b.add(docs)
    sa, ia = a.search(qs, 5)
    sb, ib = b.search(qs, 5)
    assert (ia == ib).all() and np.allclose(sa, sb, atol=1e-5)


def test_dense_retriever_and_cache(tmp_path):
    chunks = [make_chunk("sort the array of integers"), make_chunk("open a socket connection")]
    emb = CachedEmbedder(FakeEmbedder(), tmp_path)
    r = DenseRetriever(emb)
    r.index(chunks)
    assert list((tmp_path / "embeddings" / "fake").glob("d_*.npy"))
    hits = r.search(AnalyzedQuery.plain("sort integers"), 2)
    assert hits[0].chunk == chunks[0] and "semantic" in hits[0].scores
    assert r.doc_vectors([chunks[1].chunk_hash]).shape == (1, 64)


def test_cache_embeds_only_new_texts(tmp_path):
    class Counting(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.seen: list[str] = []

        def encode_documents(self, texts):
            self.seen += list(texts)
            return self._enc(texts)

    inner = Counting()
    emb = CachedEmbedder(inner, tmp_path)
    v1 = emb.encode_documents(["def a(): pass", "def b(): pass"])
    v2 = emb.encode_documents(["def a(): pass", "def b(): return 1", "def b(): pass"])
    assert inner.seen == ["def a(): pass", "def b(): pass", "def b(): return 1"]
    assert np.allclose(v2[0], v1[0]) and np.allclose(v2[2], v1[1])
    assert np.allclose(v2[1], FakeEmbedder()._enc(["def b(): return 1"])[0])


def test_local_models_disabled_guard(tmp_path):
    from seraph.config import EMBEDDERS
    from seraph.retrieval.reranker import load_reranker
    from seraph.retrieval.semantic import LazyLocalEmbedder, LocalModelsDisabledError

    e = CachedEmbedder(LazyLocalEmbedder(EMBEDDERS["qwen3-emb-0.6b"], allow_load=False), tmp_path)
    with pytest.raises(LocalModelsDisabledError):
        e.encode_documents(["not cached"] * 10)
    with pytest.raises(LocalModelsDisabledError):
        load_reranker("bge-reranker-v2-m3", allow_local=False)


@pytest.mark.gpu
def test_real_embedder_smoke():
    pytest.importorskip("sentence_transformers")
    from seraph.retrieval.semantic import load_embedder

    e = load_embedder("gte-modernbert")
    v = e.encode_queries(["reverse a linked list"])
    assert v.shape == (1, e.dim)


def test_cache_keys_use_revision_known_after_model_load(tmp_path):
    import json

    import numpy as np

    from seraph.retrieval.semantic import CachedEmbedder

    class Lazy:
        name = "lazy"
        revision = "unknown"
        dim = 2

        def encode_documents(self, texts):
            self.revision = "abc123"
            return np.ones((len(texts), 2), dtype=np.float32)

        def encode_queries(self, texts):
            return self.encode_documents(texts)

    cached = CachedEmbedder(Lazy(), tmp_path, cache_queries=True)
    cached.encode_documents(["x", "y"])
    keys = next((tmp_path / "embeddings" / "lazy").glob("d_*.keys.json"))
    assert json.loads(keys.read_text())["salt"].startswith("d|abc123|")
