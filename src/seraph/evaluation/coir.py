"""CoIR AppsRetrieval harness.

Splits:
  - `fit`:  train queries minus dev (used to train learned components)
  - `dev`:  fixed-seed 10% of train (used for all tuning)
  - `test`: official split, only reachable with `final=True`
The corpus (8,765 solutions) is shared by every split.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seraph.evaluation.metrics import Run, evaluate, latency_stats
from seraph.types import AnalyzedQuery, Chunk, ScoredChunk

TASK_NAME = "AppsRetrieval"
DEV_FRACTION = 0.1
DEV_SEED = 13


class SplitGuardError(RuntimeError):
    pass


@dataclass
class RetrievalSplit:
    name: str
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


@dataclass
class AppsData:
    corpus: dict[str, str]
    train_queries: dict[str, str]
    train_qrels: dict[str, dict[str, int]]
    test_queries: dict[str, str]
    test_qrels: dict[str, dict[str, int]]
    revision: str

    def split(self, name: str, final: bool = False) -> RetrievalSplit:
        if name == "test":
            if not final:
                raise SplitGuardError(
                    "AppsRetrieval test split is reserved for final numbers; pass final=True / --final"
                )
            return RetrievalSplit("test", self.test_queries, self.test_qrels)
        dev_ids = dev_query_ids(self.train_queries)
        if name == "dev":
            ids = sorted(dev_ids)
        elif name == "fit":
            ids = sorted(set(self.train_queries) - dev_ids)
        else:
            raise ValueError(f"unknown split {name!r}")
        return RetrievalSplit(
            name,
            {q: self.train_queries[q] for q in ids},
            {q: self.train_qrels[q] for q in ids},
        )

    def chunks(self) -> list[Chunk]:
        return corpus_to_chunks(self.corpus)


def dev_query_ids(train_queries: dict[str, str]) -> set[str]:
    ids = sorted(train_queries)
    rng = random.Random(DEV_SEED)
    rng.shuffle(ids)
    return set(ids[: int(len(ids) * DEV_FRACTION)])


def corpus_to_chunks(corpus: dict[str, str]) -> list[Chunk]:
    """Each solution becomes a `snippet` chunk; the doc id doubles as chunk hash."""
    out = []
    for doc_id, text in corpus.items():
        text = text.strip()
        out.append(Chunk(doc_id, text, "python", doc_id, None, "snippet", 1, text.count("\n") + 1))
    return out


def load_apps(cache_dir: str | Path = ".seraph_cache") -> AppsData:
    cache = Path(cache_dir) / "apps_retrieval.json"
    if cache.exists():
        return AppsData(**json.loads(cache.read_text(encoding="utf-8")))

    import mteb

    task = mteb.get_task(TASK_NAME)
    task.load_data(eval_splits=["train", "test"])
    data = AppsData(
        corpus=dict(task.corpus["test"]),
        train_queries=dict(task.queries["train"]),
        train_qrels={q: dict(r) for q, r in task.relevant_docs["train"].items()},
        test_queries=dict(task.queries["test"]),
        test_qrels={q: dict(r) for q, r in task.relevant_docs["test"].items()},
        revision=task.metadata.dataset["revision"],
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data.__dict__), encoding="utf-8")
    return data


SearchFn = Callable[[Sequence[AnalyzedQuery], int], list[list[ScoredChunk]]]


def run_search(
    search_fn: SearchFn,
    queries: dict[str, str],
    k: int = 100,
    batch_size: int = 64,
) -> tuple[dict[str, dict[str, float]], list[float]]:
    """Run batched search; returns the run and per-query latency (batch time / batch size)."""
    run: dict[str, dict[str, float]] = {}
    latencies: list[float] = []
    qids = list(queries)
    for i in range(0, len(qids), batch_size):
        batch = qids[i : i + batch_size]
        t0 = time.perf_counter()
        results = search_fn([AnalyzedQuery.plain(queries[q], qid=q) for q in batch], k)
        dt = (time.perf_counter() - t0) * 1000 / len(batch)
        for q, hits in zip(batch, results, strict=True):
            run[q] = {h.chunk_hash: float(h.score) for h in hits}
            latencies.append(dt)
    return run, latencies


def mteb_scores(qrels: dict[str, dict[str, int]], run: Run, ks: Sequence[int] = (1, 10, 100)) -> dict[str, float]:
    """Score a run with MTEB's own (pytrec_eval-based) evaluator."""
    from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator

    run = {q: dict(d) for q, d in run.items()}
    ndcg, _map, recall, _p, _naucs = RetrievalEvaluator.evaluate(qrels, run, list(ks))
    mrr, _ = RetrievalEvaluator.evaluate_custom(qrels, run, list(ks), "mrr")
    out = {}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg[f"NDCG@{k}"]
        out[f"recall@{k}"] = recall[f"Recall@{k}"]
        out[f"mrr@{k}"] = mrr[f"MRR@{k}"]
    return out


def score_run(
    run: Run,
    qrels: dict[str, dict[str, int]],
    latencies: Sequence[float] = (),
    ks: Sequence[int] = (1, 10, 100),
) -> dict[str, Any]:
    m = evaluate(run, qrels, ks)
    assert isinstance(m, dict)
    return {**m, **latency_stats(latencies)}


class SeraphMTEBModel:
    """Adapter so `mteb.MTEB(tasks=[...]).run(model)` drives Seraph's own search.

    mteb 1.x routes models whose `mteb_model_meta.name == "bm25s"` to `model.search(...)`
    instead of encoding; that is the only hook for custom first-stage retrieval there.
    """

    def __init__(self, build_search: Callable[[list[Chunk]], SearchFn], name: str = "seraph") -> None:
        from mteb.model_meta import ModelMeta

        self.build_search = build_search
        self.mteb_model_meta = ModelMeta(
            name="bm25s",
            revision=name,
            release_date=None,
            languages=None,
            loader=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            similarity_fn_name=None,
            use_instructions=False,
            training_datasets=None,
        )

    def search(
        self, corpus: dict[str, dict[str, str]], queries: dict[str, str], top_k: int, **_: Any
    ) -> dict[str, dict[str, float]]:
        texts = {d: (v.get("title", "") + " " + v["text"]) if isinstance(v, dict) else v for d, v in corpus.items()}
        search_fn = self.build_search(corpus_to_chunks(texts))
        run, _ = run_search(search_fn, queries, top_k)
        return run

    def encode(self, *args: Any, **kwargs: Any):  # pragma: no cover - never used for bm25s path
        raise NotImplementedError
