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
