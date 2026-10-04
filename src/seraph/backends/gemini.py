"""Gemini API backends: embeddings and an LLM listwise reranker."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from seraph.backends.common import KVCache, RateLimiter, require_key, with_retries
from seraph.config import ModelSpec
from seraph.retrieval.reranker import head_tail
from seraph.retrieval.vectors import normalize
from seraph.types import ScoredChunk

log = logging.getLogger(__name__)


def make_client(api_key_env: str = "GEMINI_API_KEY", client=None):
    if client is not None:
        return client
    from google import genai

    return genai.Client(api_key=require_key(api_key_env))


class GeminiEmbedder:
    """`gemini-embedding-001` with task types; every text is cached individually."""

    def __init__(self, spec: ModelSpec, cache_dir: str | Path, client=None) -> None:
        o = spec.options
        self.spec = spec
        self.name = spec.name
        self.model = spec.hf_id
        self.dim = int(o.get("output_dim", 768))
        self.revision = f"{self.model}-d{self.dim}"
        self.query_task = o.get("query_task", "CODE_RETRIEVAL_QUERY")
        self.doc_task = o.get("doc_task", "RETRIEVAL_DOCUMENT")
        self.batch = int(o.get("batch", 100))
        self.max_chars = int(o.get("max_chars", 6000))
        self._client = client
        self._limiter = RateLimiter(float(o.get("rpm", 90)))
        self._cache = KVCache(Path(cache_dir) / "api" / f"{spec.name}.sqlite")

    @property
    def client(self):
        self._client = make_client(self.spec.options.get("api_key_env", "GEMINI_API_KEY"), self._client)
        return self._client

    def _call(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import types

        def go():
            self._limiter.wait()
            resp = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
            )
            return np.array([e.values for e in resp.embeddings], dtype=np.float32)

        return with_retries(go)

    def _encode(self, texts: Sequence[str], task: str, prefix: str) -> np.ndarray:
        texts = [(prefix + t)[: self.max_chars] or " " for t in texts]
        keys = [KVCache.key(self.revision, task, t) for t in texts]
        have = self._cache.get_many(list(set(keys)))
        missing = sorted({k: t for k, t in zip(keys, texts, strict=True) if k not in have}.items())
        for i in range(0, len(missing), self.batch):
            part = missing[i : i + self.batch]
            vecs = self._call([t for _, t in part], task)
            new = {k: v.tobytes() for (k, _), v in zip(part, vecs, strict=True)}
            self._cache.put_many(new)
            have.update(new)
            if len(missing) > self.batch:
                log.info("embedded %d/%d", min(i + self.batch, len(missing)), len(missing))
        return normalize(np.stack([np.frombuffer(have[k], dtype=np.float32) for k in keys]))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_task, self.spec.query_prefix)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.doc_task, self.spec.doc_prefix)


LISTWISE_PROMPT = """You rank candidate code snippets by how likely each one solves the programming task.

Task:
{query}

Candidates:
{candidates}

Return ONLY a JSON array of the candidate numbers, best first, including every candidate once."""


class GeminiListwiseReranker:
    """One Gemini call per query reorders the top-N candidates; results are cached per query+set."""

    def __init__(self, spec: ModelSpec, cache_dir: str | Path, blend: float = 0.0, client=None) -> None:
        o = spec.options
        self.spec = spec
        self.name = spec.name
        self.model = spec.hf_id
        self.revision = self.model
        self.blend = blend
        self.window = int(o.get("window", 20))
        self.doc_chars = int(o.get("doc_chars", 1200))
        self.query_chars = int(o.get("query_chars", 3000))
        self._client = client
        self._limiter = RateLimiter(float(o.get("rpm", 14)))
        self._cache = KVCache(Path(cache_dir) / "api" / f"{spec.name}.sqlite")

    @property
    def client(self):
        self._client = make_client(self.spec.options.get("api_key_env", "GEMINI_API_KEY"), self._client)
        return self._client

    def _prompt(self, query: str, cands: Sequence[ScoredChunk]) -> str:
        body = "\n\n".join(
            f"[{i + 1}]\n```\n{head_tail(c.chunk.text, self.doc_chars)}\n```" for i, c in enumerate(cands)
        )
        return LISTWISE_PROMPT.format(query=head_tail(query, self.query_chars), candidates=body)

    def _order(self, query: str, cands: Sequence[ScoredChunk]) -> list[int]:
        key = KVCache.key(self.model, query, *[c.chunk_hash for c in cands])
        hit = self._cache.get_many([key])
        if key in hit:
            return json.loads(hit[key])
        from google.genai import types

        def go():
            self._limiter.wait()
            return self.client.models.generate_content(
                model=self.model,
                contents=self._prompt(query, cands),
                config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
            ).text

        order = parse_order(with_retries(go) or "", len(cands))
        self._cache.put_many({key: json.dumps(order).encode()})
        return order

    def rerank_batch(
        self, queries: Sequence[str], candidates: Sequence[Sequence[ScoredChunk]], top_k: int
    ) -> list[list[ScoredChunk]]:
        out = []
        for q, cands in zip(queries, candidates, strict=True):
            head, rest = list(cands[: self.window]), list(cands[self.window :])
            order = self._order(q, head) if head else []
            n = len(head)
            ranked = []
            for rank, idx in enumerate(order):
                c = head[idx]
                s = 1.0 - rank / max(n, 1)
                final = s if self.blend == 0 else (1 - self.blend) * s + self.blend * c.score
                ranked.append(ScoredChunk(c.chunk, final, {**c.scores, "rerank": s}))
            ranked.sort(key=lambda x: -x.score)
            floor = min((r.score for r in ranked), default=0.0)
            ranked += [ScoredChunk(c.chunk, floor - (j + 1) * 1e-3, c.scores) for j, c in enumerate(rest)]
            out.append(ranked[:top_k])
        return out

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return self.rerank_batch([query], [candidates], top_k)[0]


def parse_order(text: str, n: int) -> list[int]:
    """Parse a 1-based ranking; drop invalid/duplicate ids and append any the model skipped."""
    try:
        raw = json.loads(text)
        nums = raw if isinstance(raw, list) else raw.get("ranking", [])
    except (json.JSONDecodeError, AttributeError):
        nums = re.findall(r"\d+", text)
    seen: list[int] = []
    for x in nums:
        try:
            i = int(x) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            seen.append(i)
    return seen + [i for i in range(n) if i not in seen]
