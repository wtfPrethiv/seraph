"""In-memory fakes for every cross-owner protocol. Keep these in sync with protocols.py."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

import numpy as np

from seraph.memory import InMemoryChunkStore, InMemoryCodeGraph, InMemoryVersionStore
from seraph.retrieval.base import BaseRetriever
from seraph.types import (
    AnalyzedQuery,
    ChangeType,
    Chunk,
    EdgeKind,
    LineageNode,
    ScoredChunk,
    Version,
)

__all__ = [
    "FakeCodeGraph",
    "FakeEmbedder",
    "FakeRetriever",
    "FakeVersionStore",
    "InMemoryChunkStore",
    "make_chunk",
    "sample_repo",
]

FakeCodeGraph = InMemoryCodeGraph
FakeVersionStore = InMemoryVersionStore


def make_chunk(
    text: str,
    symbol: str | None = None,
    file: str = "a.py",
    version_id: str = "HEAD",
    lineage_id: str | None = None,
    kind: str = "function",
) -> Chunk:
    h = hashlib.sha1(f"{text}".encode()).hexdigest()[:16]
    return Chunk(h, text, "python", file, symbol, kind, 1, text.count("\n") + 1, version_id, lineage_id)


class FakeEmbedder:
    """Deterministic hashed bag-of-tokens embedder."""

    name = "fake"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _enc(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)

    encode_queries = _enc
    encode_documents = _enc


class FakeRetriever(BaseRetriever):
    """Returns chunks containing any query token, scored by overlap count."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self._chunks: list[Chunk] = []

    def index(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)

    def search(self, query: AnalyzedQuery, k: int, version: str = "HEAD") -> list[ScoredChunk]:
        toks = set(re.findall(r"\w+", query.text.lower()))
        hits = []
        for c in self._chunks:
            s = len(toks & set(re.findall(r"\w+", c.text.lower())))
            if s:
                hits.append(ScoredChunk(c, float(s), {self.name: float(s)}))
        hits.sort(key=lambda x: -x.score)
        return hits[:k]


def sample_repo() -> tuple[InMemoryChunkStore, InMemoryCodeGraph, InMemoryVersionStore]:
    """Tiny repo: parse_config -> read_file, load -> parse_config, with two versions of parse_config."""
    v1 = Version("v1", "aaa", "v1.0", None, 1)
    v2 = Version("v2", "bbb", "v2.0", "v1", 2)
    read_file = make_chunk("def read_file(path):\n    return open(path).read()", "io.read_file", "io.py", "HEAD", "L_read")
    parse_old = make_chunk("def parse_config(path):\n    return json.loads(read_file(path))", "cfg.parse_config", "cfg.py", "v1", "L_parse")
    parse_new = make_chunk("def parse_config(path, strict=False):\n    data = json.loads(read_file(path))\n    validate(data, strict)\n    return data", "cfg.parse_config", "cfg.py", "HEAD", "L_parse")
    load = make_chunk("def load(app):\n    app.config = parse_config(app.path)", "app.load", "app.py", "HEAD", "L_load")
    unrelated = make_chunk("def add(a, b):\n    return a + b", "math.add", "math.py", "HEAD", "L_add")

    store = InMemoryChunkStore([read_file, parse_old, parse_new, load, unrelated])
    graph = InMemoryCodeGraph()
    for c in (read_file, parse_new, load, unrelated):
        graph.add_symbol(c.symbol, c.chunk_hash)  # type: ignore[arg-type]
    graph.add_edge("cfg.parse_config", "io.read_file", EdgeKind.CALLS)
    graph.add_edge("app.load", "cfg.parse_config", EdgeKind.CALLS)

    versions = InMemoryVersionStore([v1, v2])
    versions.add_lineage_node(LineageNode("L_parse", "cfg.parse_config", "v1", parse_old.chunk_hash, ChangeType.ADDED, "add config parser"), parse_old.text)
    versions.add_lineage_node(LineageNode("L_parse", "cfg.parse_config", "v2", parse_new.chunk_hash, ChangeType.MODIFIED, "add strict validation to config parsing"), parse_new.text)
    return store, graph, versions
