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


if __name__ == "__main__":
    app()
