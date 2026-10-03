"""`seraph-eval` command line: AppsRetrieval experiments."""

from __future__ import annotations

import json
import random

import typer

from seraph.evaluation.coir import load_apps, mteb_scores
from seraph.evaluation.metrics import evaluate

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Seraph evaluation commands."""


@app.command()
def crosscheck(split: str = "dev", cache_dir: str = ".seraph_cache", seed: int = 0) -> None:
    """Score a random run with both MTEB and seraph.metrics; fail if they differ by >1e-4."""
    data = load_apps(cache_dir)
    s = data.split(split)
    rng = random.Random(seed)
    docs = list(data.corpus)
    run = {}
    for q, rels in s.qrels.items():
        cand = rng.sample(docs, 99) + list(rels)
        run[q] = {d: rng.random() for d in cand}
    ours = evaluate(run, s.qrels)
    theirs = mteb_scores(s.qrels, run)
    diff = {k: abs(ours[k] - v) for k, v in theirs.items()}
    typer.echo(json.dumps({"ours": ours, "mteb": theirs, "max_abs_diff": max(diff.values())}, indent=2))
    if max(diff.values()) > 1e-4:
        raise typer.Exit(1)


@app.command()
def run(
    config: list[str],
    split: str = "dev",
    final: bool = typer.Option(False, "--final", help="Allow the AppsRetrieval test split"),
    max_queries: int | None = None,
    set_: list[str] = typer.Option([], "--set", help="override, e.g. retrieval.bm25_k1=0.9"),
) -> None:
    """Run one or more experiment configs and print their metrics."""
    from seraph.config import load_config
    from seraph.evaluation.experiment import run_experiment

    for path in config:
        cfg = load_config(path, _parse_overrides(set_))
        res = run_experiment(cfg, split=split, final=final, max_queries=max_queries)
        m = res["metrics"]
        typer.echo(
            f"{cfg.name:32s} ndcg@10={m['ndcg@10']:.4f} mrr@10={m['mrr@10']:.4f} "
            f"r@10={m['recall@10']:.4f} r@100={m['recall@100']:.4f} "
            f"p50={m.get('latency_p50_ms', 0):.1f}ms"
        )


@app.command()
def sweep(spec: str = "experiments/configs/dense_sweep.yaml", only: list[str] = typer.Option([])) -> None:
    """Dense model sweep; one model at a time, freeing GPU memory in between."""
    import gc
    import traceback
    from pathlib import Path

    import yaml

    from seraph.config import load_config
    from seraph.evaluation.experiment import run_experiment

    s = yaml.safe_load(Path(spec).read_text())
    base = Path(spec).parent / s["base"]
    for model in s["models"]:
        if only and model not in only:
            continue
        for split in s["splits"]:
            cfg = load_config(base, {"name": f"b2_dense_{model}", "retrieval": {"dense_model": model}})
            try:
                m = run_experiment(cfg, split=split)["metrics"]
                typer.echo(f"{model:18s} {split:10s} ndcg@10={m['ndcg@10']:.4f} r@100={m['recall@100']:.4f}")
            except Exception:
                typer.echo(f"{model:18s} {split:10s} FAILED")
                traceback.print_exc()
                break
            finally:
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass


@app.command("table")
def table_cmd(
    prefix: str = "",
    splits: list[str] = typer.Option(["dev", "dev_stdin"], "--split"),
    out: str | None = None,
    results_dir: str = "experiments/results",
) -> None:
    """Print (and optionally save) a markdown results table."""
    from pathlib import Path

    from seraph.evaluation.report import load_results, table

    md = table(load_results(results_dir, prefix), splits)
    typer.echo(md)
    if out:
        Path(out).write_text(md)


@app.command("tune-fusion")
def tune_fusion(config: str = "experiments/configs/b3_hybrid_weighted.yaml", split: str = "dev_stdin") -> None:
    """Retrieve each view once, then grid-search RRF k and weighted-fusion weights."""
    from seraph.config import load_config
    from seraph.evaluation.coir import score_run
    from seraph.memory import InMemoryChunkStore
    from seraph.retrieval.fusion import fuse
    from seraph.retrieval.pipeline import Pipeline
    from seraph.types import AnalyzedQuery

    cfg = load_config(config)
    data = load_apps(cfg.cache_dir)
    s = data.split(split)
    pipe = Pipeline.from_config(cfg, InMemoryChunkStore(data.chunks()))
    qs = [AnalyzedQuery.plain(t, qid=q) for q, t in s.queries.items()]
    per_view = pipe.retrieve_views(qs, cfg.retrieval.first_stage_k)

    def score(method, weights=None, rrf_k=60, norm="minmax"):
        run_ = {}
        for i, q in enumerate(qs):
            hits = fuse({v: h[i] for v, h in per_view.items()}, method, weights, rrf_k, norm)
            run_[q.qid] = {h.chunk_hash: h.score for h in hits[:100]}
        return score_run(run_, s.qrels)

    rows = []
    for k in (10, 30, 60, 100):
        m = score("rrf", rrf_k=k)
        rows.append((m["ndcg@10"], m["recall@100"], f"rrf k={k}"))
    for norm in ("minmax", "zscore"):
        for wl in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            m = score("weighted", {"lexical": wl, "semantic": 1 - wl}, norm=norm)
            rows.append((m["ndcg@10"], m["recall@100"], f"weighted {norm} lexical={wl}"))
    rows.sort(reverse=True)
    for ndcg, r100, desc in rows:
        typer.echo(f"ndcg@10={ndcg:.4f} r@100={r100:.4f} {desc}")


@app.command("tune-bm25")
def tune_bm25(split: str = "dev_stdin", cache_dir: str = ".seraph_cache") -> None:
    """Grid-search BM25 k1/b/stemming on a tuning split."""
    from seraph.evaluation.coir import run_search, score_run
    from seraph.retrieval.lexical import LexicalRetriever

    data = load_apps(cache_dir)
    s = data.split(split)
    chunks = data.chunks()
    rows = []
    for stem in (True, False):
        for k1 in (0.6, 0.9, 1.2, 1.5, 2.0):
            for b in (0.3, 0.5, 0.75, 0.9):
                r = LexicalRetriever(k1, b, stem)
                r.index(chunks)
                run_, _ = run_search(r.search_batch, s.queries, 100)
                m = score_run(run_, s.qrels)
                rows.append((m["ndcg@10"], m["recall@100"], k1, b, stem))
    rows.sort(reverse=True)
    for ndcg, r100, k1, b, stem in rows[:8]:
        typer.echo(f"ndcg@10={ndcg:.4f} r@100={r100:.4f} k1={k1} b={b} stem={stem}")


def _parse_overrides(items: list[str]) -> dict:
    import yaml

    out: dict = {}
    for item in items:
        key, val = item.split("=", 1)
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return out


if __name__ == "__main__":
    app()
