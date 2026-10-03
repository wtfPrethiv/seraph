"""Run one config on one AppsRetrieval split and record metrics + metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from seraph.config import SeraphConfig
from seraph.evaluation.coir import load_apps, run_search, score_run
from seraph.evaluation.runfiles import write_run
from seraph.memory import InMemoryChunkStore

RESULTS_DIR = Path("experiments/results")
RUNS_DIR = Path("experiments/runs")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def config_hash(cfg: SeraphConfig) -> str:
    return hashlib.sha1(cfg.model_dump_json().encode()).hexdigest()[:10]


def run_experiment(
    cfg: SeraphConfig,
    split: str = "dev",
    final: bool = False,
    k: int = 100,
    results_dir: Path = RESULTS_DIR,
    max_queries: int | None = None,
) -> dict[str, Any]:
    from seraph.retrieval.pipeline import Pipeline

    set_seed(cfg.seed)
    data = load_apps(cfg.cache_dir)
    s = data.split(split, final=final)
    queries = s.queries
    if max_queries:
        queries = dict(list(queries.items())[:max_queries])
    qrels = {q: s.qrels[q] for q in queries}

    t0 = time.perf_counter()
    pipe = Pipeline.from_config(cfg, InMemoryChunkStore(data.chunks()))
    index_s = time.perf_counter() - t0

    run, lat = run_search(pipe.search_batch, queries, k)
    metrics = score_run(run, qrels, lat)
    result = {
        "name": cfg.name,
        "split": split,
        "n_queries": len(queries),
        "metrics": metrics,
        "index_seconds": index_s,
        "meta": {
            "git_sha": git_sha(),
            "config_hash": config_hash(cfg),
            "dataset_revision": data.revision,
            "models": pipe.model_metadata(),
            "seed": cfg.seed,
            "python": platform.python_version(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "config": cfg.model_dump(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{cfg.name}__{split}.json").write_text(json.dumps(result, indent=2))
    write_run(run, RUNS_DIR / f"{cfg.name}__{split}.trec", tag=cfg.name)
    return result
