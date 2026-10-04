"""CoIR AppsRetrieval harness.

Splits:
  - `fit`:  train queries minus dev (used to train learned components)
  - `dev`:  fixed-seed 20% of train (used for all tuning)
  - `dev_stdin`: dev queries whose gold solution reads stdin
  - `test`: official split, only reachable with `final=True`
The corpus (8,765 solutions) is shared by every split.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seraph.evaluation.metrics import Run, evaluate, latency_stats
from seraph.types import AnalyzedQuery, Chunk, ScoredChunk

TASK_NAME = "AppsRetrieval"
DEV_FRACTION = 0.2
DEV_SEED = 13
# Train golds are ~1/3 stdin programs (rest call-based); `dev_stdin` mirrors the stdin format.
_STDIN = re.compile(r"\binput\s*\(|sys\.stdin")


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
        elif name == "dev_stdin":
            ids = sorted(q for q in dev_ids if self._gold_reads_stdin(q))
        elif name == "fit":
            ids = sorted(set(self.train_queries) - dev_ids)
        else:
            raise ValueError(f"unknown split {name!r}")
        return RetrievalSplit(
            name,
            {q: self.train_queries[q] for q in ids},
            {q: self.train_qrels[q] for q in ids},
        )

    def _gold_reads_stdin(self, qid: str) -> bool:
        return any(_STDIN.search(self.corpus[d]) for d in self.train_qrels[qid])

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

    task = mteb.get_task(TASK_NAME, eval_splits=["train", "test"])
    task.load_data()
    splits = task.dataset["default"]
    data = AppsData(
        corpus={r["id"]: r["text"] for r in splits["test"]["corpus"]},
        train_queries={r["id"]: r["text"] for r in splits["train"]["queries"]},
        train_qrels={q: dict(r) for q, r in splits["train"]["relevant_docs"].items()},
        test_queries={r["id"]: r["text"] for r in splits["test"]["queries"]},
        test_qrels={q: dict(r) for q, r in splits["test"]["relevant_docs"].items()},
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
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

    run = {q: dict(d) for q, d in run.items()}
    res = calculate_retrieval_scores(run, qrels, list(ks))
    out = {}
    for k in ks:
        out[f"ndcg@{k}"] = res.ndcg[f"NDCG@{k}"]
        out[f"recall@{k}"] = res.recall[f"Recall@{k}"]
        out[f"mrr@{k}"] = res.mrr[f"MRR@{k}"]
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


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(p for p in (row.get("title") or "", row["text"]) if p)


class SeraphMTEBModel:
    """mteb 2.x `SearchProtocol` model: `mteb.evaluate` hands it the corpus and queries, and the
    whole Seraph pipeline (every view, fusion, rerank) produces the ranking.

    It must not define `encode`, or mteb would wrap it as a plain embedding model instead.
    """

    def __init__(
        self,
        build_search: Callable[[list[Chunk]], SearchFn],
        name: str = "seraph",
        revision: str = "dev",
        batch_size: int = 64,
    ) -> None:
        from mteb.models.model_meta import ModelMeta

        self.build_search = build_search
        self.batch_size = batch_size
        self._search: SearchFn | None = None
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=f"seraph/{name}",
            revision=revision,
            release_date=None,
            languages=["eng-Latn", "python-Code"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license="mit",
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            reference=None,
            similarity_fn_name=None,
            use_instructions=True,
            training_datasets=None,
        )

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc=None) -> None:
        self._search = self.build_search(corpus_to_chunks({r["id"]: _row_text(r) for r in corpus}))

    def search(
        self, queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs, top_ranked=None, num_proc=None
    ) -> dict[str, dict[str, float]]:
        if self._search is None:
            raise RuntimeError("index() must be called before search()")
        run, _ = run_search(self._search, {r["id"]: r["text"] for r in queries}, top_k, self.batch_size)
        if top_ranked:
            allowed = {q: set(top_ranked.get(q, hits)) for q, hits in run.items()}
            run = {q: {d: s for d, s in hits.items() if d in allowed[q]} for q, hits in run.items()}
        return run
