"""Dense view (beta): pluggable embedders, embedding cache, vector search."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from seraph.config import EMBEDDERS, ModelSpec, resolve_device
from seraph.protocols import Embedder
from seraph.retrieval.base import BaseRetriever
from seraph.retrieval.vectors import VectorIndex, normalize
from seraph.types import AnalyzedQuery, Chunk, ScoredChunk, View

log = logging.getLogger(__name__)


def _transformers_compat() -> None:
    """Older remote-code models (CodeSage) import Conv1D from its pre-4.4x location."""
    import transformers.modeling_utils as mu

    if not hasattr(mu, "Conv1D"):
        from transformers.pytorch_utils import Conv1D

        mu.Conv1D = Conv1D


def _torch_dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


class STEmbedder:
    """sentence-transformers backend."""

    def __init__(self, spec: ModelSpec, device: str = "auto") -> None:
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.name = spec.name
        device = resolve_device(device)
        kwargs = {"dtype": _torch_dtype(spec.dtype)} if device == "cuda" else {}
        self.model = SentenceTransformer(
            spec.hf_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            device=device,
            model_kwargs=kwargs,
        )
        self.model.max_seq_length = spec.max_len
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.revision = _commit_hash(self.model[0].auto_model)

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.spec.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        ).astype(np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.spec.query_prefix)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.spec.doc_prefix)


class HFEmbedder:
    """Raw transformers encoder with mean/CLS pooling (CodeBERT, UniXcoder, CodeT5+)."""

    def __init__(self, spec: ModelSpec, device: str = "auto") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.spec = spec
        self.name = spec.name
        self.device = resolve_device(device)
        self.tok = AutoTokenizer.from_pretrained(
            spec.hf_id, revision=spec.revision, trust_remote_code=spec.trust_remote_code
        )
        dtype = _torch_dtype(spec.dtype) if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            spec.hf_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            dtype=dtype,
        ).to(self.device).eval()
        self.revision = _commit_hash(self.model)
        self.dim = int(self._forward(["x"]).shape[1])

    def _forward(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        enc = self.tok(
            list(texts), padding=True, truncation=True, max_length=self.spec.max_len, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            if self.spec.backend == "codet5p":
                emb = self.model(enc["input_ids"], attention_mask=enc["attention_mask"])
            else:
                hidden = self.model(**enc).last_hidden_state
                if self.spec.pooling == "cls":
                    emb = hidden[:, 0]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return emb.float().cpu().numpy()

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        texts = [prefix + t for t in texts]
        order = np.argsort([-len(t) for t in texts])  # length-sorted batches waste less padding
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        bs = self.spec.batch_size
        for i in range(0, len(texts), bs):
            idx = order[i : i + bs]
            out[idx] = self._forward([texts[j] for j in idx])
        return normalize(out)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.spec.query_prefix)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.spec.doc_prefix)


def _commit_hash(model) -> str:
    return getattr(getattr(model, "config", None), "_commit_hash", None) or "unknown"


def _load_local(spec: ModelSpec, device: str) -> Embedder:
    if spec.trust_remote_code:
        _transformers_compat()
    if spec.backend == "st":
        return STEmbedder(spec, device)
    return HFEmbedder(spec, device)


def _hf_cached_revision(spec: ModelSpec) -> str | None:
    """Commit hash of an already-downloaded model, read from the HF cache without loading it."""
    import os

    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    ref = home / f"models--{spec.hf_id.replace('/', '--')}" / "refs" / spec.revision
    return ref.read_text().strip() if ref.exists() else None


class LazyLocalEmbedder:
    """Defers loading a local model until a text is not found in the embedding cache."""

    def __init__(self, spec: ModelSpec, device: str = "auto") -> None:
        self.spec = spec
        self.name = spec.name
        self.device = device
        self._model: Embedder | None = None
        self.revision = _hf_cached_revision(spec) or "unknown"

    @property
    def model(self) -> Embedder:
        if self._model is None:
            log.info("loading local embedder %s", self.spec.hf_id)
            self._model = _load_local(self.spec, self.device)
            self.revision = getattr(self._model, "revision", self.revision)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.dim

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode_queries(texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode_documents(texts)


def load_embedder(
    name: str,
    device: str = "auto",
    cache_dir: str | Path = ".seraph_cache",
    fallback: str | None = None,
) -> Embedder:
    """API models first; `fallback` (usually local) is used only when the API key is missing."""
    import os

    spec = EMBEDDERS[name]
    if spec.backend == "gemini_embed":
        from seraph.backends.gemini import GeminiEmbedder

        if fallback and not os.environ.get(spec.options.get("api_key_env", "GEMINI_API_KEY")):
            log.warning("no API key for %s; falling back to %s", name, fallback)
            return load_embedder(fallback, device, cache_dir)
        return GeminiEmbedder(spec, cache_dir)
    return LazyLocalEmbedder(spec, device)


class CachedEmbedder:
    """Wraps an embedder; caches matrices on disk keyed by model, prefix, max_len and text content."""

    def __init__(
        self,
        inner: Embedder,
        cache_dir: str | Path,
        spec: ModelSpec | None = None,
        cache_queries: bool = True,
    ) -> None:
        self.inner = inner
        self.cache_queries = cache_queries
        self.name = inner.name
        self.revision = getattr(inner, "revision", "unknown")
        self.spec = spec
        self.dir = Path(cache_dir) / "embeddings" / inner.name
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def dim(self) -> int:
        return self.inner.dim

    def _key(self, kind: str, texts: Sequence[str]) -> Path:
        h = hashlib.sha1()
        h.update(f"{kind}|{self.revision}|".encode())
        if self.spec:
            h.update(f"{self.spec.query_prefix}|{self.spec.doc_prefix}|{self.spec.max_len}".encode())
        for t in texts:
            h.update(hashlib.sha1(t.encode()).digest())
        return self.dir / f"{kind}_{h.hexdigest()[:20]}.npy"

    def _cached(self, kind: str, texts: Sequence[str], fn) -> np.ndarray:
        path = self._key(kind, texts)
        if path.exists():
            return np.load(path)
        mat = fn(texts)
        np.save(path, mat)
        return mat

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if not self.cache_queries or len(texts) < 8:  # interactive queries: not worth a cache file each
            return self.inner.encode_queries(texts)
        return self._cached("q", texts, self.inner.encode_queries)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._cached("d", texts, self.inner.encode_documents)


class DenseRetriever(BaseRetriever):
    name = View.SEMANTIC.value

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.revision = f"{embedder.name}@{getattr(embedder, 'revision', 'unknown')}"
        self._chunks: list[Chunk] = []
        self._index: VectorIndex | None = None
        self._pos: dict[str, int] = {}

    def index(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        self._pos = {c.chunk_hash: i for i, c in enumerate(self._chunks)}
        vecs = self.embedder.encode_documents([c.text for c in self._chunks])
        self._index = VectorIndex(vecs.shape[1])
        self._index.add(vecs)

    def doc_vectors(self, chunk_hashes: Sequence[str]) -> np.ndarray:
        assert self._index is not None
        return self._index.matrix[[self._pos[h] for h in chunk_hashes]]

    def encode_queries(self, queries: Sequence[AnalyzedQuery]) -> np.ndarray:
        return self.embedder.encode_queries([q.text for q in queries])

    def search_batch(
        self, queries: Sequence[AnalyzedQuery], k: int, version: str = "HEAD"
    ) -> list[list[ScoredChunk]]:
        assert self._index is not None, "index() first"
        if not queries:
            return []
        scores, idx = self._index.search(self.encode_queries(queries), k)
        return [
            [ScoredChunk(self._chunks[i], float(s), {self.name: float(s)}) for i, s in zip(ri, rs, strict=True) if i >= 0]
            for ri, rs in zip(idx, scores, strict=True)
        ]
