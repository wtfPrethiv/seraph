"""Exact inner-product vector index (FAISS when available, numpy otherwise)."""

from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class VectorIndex:
    def __init__(self, dim: int, use_faiss: bool = True) -> None:
        self.dim = dim
        self._faiss = None
        self._mat: np.ndarray | None = None
        if use_faiss:
            try:
                import faiss

                self._faiss = faiss.IndexFlatIP(dim)
            except ImportError:
                self._faiss = None

    def add(self, vectors: np.ndarray) -> None:
        v = normalize(vectors)
        if self._faiss is not None:
            self._faiss.add(v)
        self._mat = v if self._mat is None else np.vstack([self._mat, v])

    def __len__(self) -> int:
        return 0 if self._mat is None else len(self._mat)

    @property
    def matrix(self) -> np.ndarray:
        assert self._mat is not None
        return self._mat

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = normalize(queries)
        k = min(k, len(self))
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            return scores, idx
        sims = q @ self.matrix.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx
