"""Second-stage rerankers with an on-disk (query, doc) score cache."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from seraph.config import RERANKERS, ModelSpec, resolve_device
from seraph.types import ScoredChunk

RERANK_INSTRUCTION = "Given a programming problem description, judge whether the code solves it"


def head_tail(text: str, max_chars: int, head_frac: float = 0.7) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * head_frac)
    return text[:head] + "\n...\n" + text[-(max_chars - head) :]


class PairScoreCache:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.data: dict[str, float] = {}
        if path and path.exists():
            self.data = pickle.loads(path.read_bytes())

    @staticmethod
    def key(query: str, doc_hash: str) -> str:
        return hashlib.sha1(query.encode()).hexdigest()[:16] + ":" + doc_hash

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(pickle.dumps(self.data))


class _BaseReranker:
    name: str
    spec: ModelSpec

    def __init__(self, spec: ModelSpec, cache_dir: str | Path | None = None, blend: float = 0.0) -> None:
        self.spec = spec
        self.name = spec.name
        self.blend = blend
        self.cache = PairScoreCache(Path(cache_dir) / "rerank" / f"{spec.name}.pkl" if cache_dir else None)
        self.query_chars = 3000
        self.doc_chars = 3000

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        raise NotImplementedError

    def rerank_batch(
        self, queries: Sequence[str], candidates: Sequence[Sequence[ScoredChunk]], top_k: int
    ) -> list[list[ScoredChunk]]:
        todo: list[tuple[str, str]] = []
        todo_keys: list[str] = []
        for q, cands in zip(queries, candidates, strict=True):
            for c in cands:
                key = PairScoreCache.key(q, c.chunk_hash)
                if key not in self.cache.data and key not in todo_keys:
                    todo.append((head_tail(q, self.query_chars), head_tail(c.chunk.text, self.doc_chars)))
                    todo_keys.append(key)
        if todo:
            order = np.argsort([-(len(a) + len(b)) for a, b in todo])
            scores = np.zeros(len(todo))
            bs = self.spec.batch_size * 4
            for i in range(0, len(todo), bs):
                idx = order[i : i + bs]
                scores[idx] = self.score_pairs([todo[j] for j in idx])
            self.cache.data.update(zip(todo_keys, scores.tolist(), strict=True))
            self.cache.save()

        out = []
        for q, cands in zip(queries, candidates, strict=True):
            rescored = []
            for c in cands:
                s = self.cache.data[PairScoreCache.key(q, c.chunk_hash)]
                final = s if self.blend == 0 else (1 - self.blend) * s + self.blend * c.score
                rescored.append(ScoredChunk(c.chunk, final, {**c.scores, "rerank": s}))
            rescored.sort(key=lambda x: (-x.score, x.chunk_hash))
            out.append(rescored[:top_k])
        return out

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return self.rerank_batch([query], [candidates], top_k)[0]


class CrossEncoderReranker(_BaseReranker):
    def __init__(self, spec: ModelSpec, device: str = "auto", **kw) -> None:
        super().__init__(spec, **kw)
        import torch
        from sentence_transformers import CrossEncoder

        device = resolve_device(device)
        self.model = CrossEncoder(
            spec.hf_id,
            revision=spec.revision,
            max_length=spec.max_len,
            device=device,
            trust_remote_code=spec.trust_remote_code,
            model_kwargs={"dtype": torch.float16} if device == "cuda" else {},
        )
        self.revision = getattr(self.model.model.config, "_commit_hash", None) or "unknown"

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        return np.asarray(
            self.model.predict(list(pairs), batch_size=self.spec.batch_size, show_progress_bar=False),
            dtype=np.float64,
        )


class CausalLMReranker(_BaseReranker):
    """Qwen3-Reranker: P(yes) from the final-token logits of a judge prompt."""

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, spec: ModelSpec, device: str = "auto", instruction: str = RERANK_INSTRUCTION, **kw) -> None:
        super().__init__(spec, **kw)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = resolve_device(device)
        self.instruction = instruction
        self.tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            revision=spec.revision,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or "unknown"
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        self.prefix_ids = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(self.SUFFIX, add_special_tokens=False)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        import torch

        bodies = [f"<Instruct>: {self.instruction}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
        budget = self.spec.max_len - len(self.prefix_ids) - len(self.suffix_ids)
        out = np.zeros(len(pairs))
        bs = self.spec.batch_size
        for i in range(0, len(bodies), bs):
            enc = self.tok(
                bodies[i : i + bs], truncation="longest_first", max_length=budget,
                add_special_tokens=False, padding=False,
            )
            ids = [self.prefix_ids + x + self.suffix_ids for x in enc["input_ids"]]
            batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**batch).logits[:, -1, :]
            two = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1).float()
            out[i : i + bs] = torch.log_softmax(two, dim=1)[:, 1].exp().cpu().numpy()
        return out


def load_reranker(
    name: str,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    blend: float = 0.0,
    fallback: str | None = None,
):
    """API rerankers first; `fallback` (usually local) is used only when the API key is missing."""
    import os

    spec = RERANKERS[name]
    if spec.backend == "gemini_rerank":
        from seraph.backends.gemini import GeminiListwiseReranker

        if fallback and not os.environ.get(spec.options.get("api_key_env", "GEMINI_API_KEY")):
            return load_reranker(fallback, device, cache_dir, blend)
        return GeminiListwiseReranker(spec, cache_dir or ".seraph_cache", blend)
    cls = CausalLMReranker if spec.backend == "causal_reranker" else CrossEncoderReranker
    return cls(spec, device=device, cache_dir=cache_dir, blend=blend)
