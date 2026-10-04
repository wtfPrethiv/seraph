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
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("seraph").setLevel(logging.INFO)


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
def submit(
    config: str = "experiments/configs/submission.yaml",
    final: bool = typer.Option(False, "--final", help="Required: this scores the AppsRetrieval test split"),
    out: str = "appsretrieval_results.json",
    predictions: str | None = typer.Option(None, help="folder for mteb's per-query rankings"),
    batch_size: int = 64,
    set_: list[str] = typer.Option([], "--set", help="override, e.g. retrieval.rerank_depth=10"),
) -> None:
    """Run `mteb.evaluate` on AppsRetrieval (test split) with the full pipeline; write the upload JSON."""
    import time
    from pathlib import Path

    import mteb

    from seraph.config import load_config
    from seraph.evaluation.coir import TASK_NAME, SeraphMTEBModel
    from seraph.evaluation.experiment import config_hash, set_seed
    from seraph.memory import InMemoryChunkStore
    from seraph.retrieval.pipeline import Pipeline

    if not final:
        typer.echo("submit scores the test split; pass --final once the config is frozen", err=True)
        raise typer.Exit(2)
    cfg = load_config(config, _parse_overrides(set_))
    set_seed(cfg.seed)

    def build_search(chunks):
        return Pipeline.from_config(cfg, InMemoryChunkStore(chunks)).search_batch

    model = SeraphMTEBModel(build_search, cfg.name, config_hash(cfg), batch_size)
    t0 = time.perf_counter()
    result = mteb.evaluate(
        model,
        [mteb.get_task(TASK_NAME)],
        encode_kwargs={"batch_size": batch_size},
        overwrite_strategy="always",
        prediction_folder=predictions,
    )
    task_result = list(result.task_results)[0]
    task_result.to_disk(Path(out))  # to_dict() keeps a datetime that json.dump rejects
    scores = task_result.scores["test"][0]
    typer.echo(
        f"{cfg.name}: ndcg@10={scores['ndcg_at_10']:.4f} mrr@10={scores['mrr_at_10']:.4f} "
        f"({time.perf_counter() - t0:.0f}s) -> {out}"
    )


def _known_answers(data) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    for queries, qrels in ((data.train_queries, data.train_qrels), (data.test_queries, data.test_qrels)):
        for qid, text in queries.items():
            known[" ".join(text.split())] = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
    return known


def _read_problem() -> str | None:
    typer.echo(typer.style("\nPaste a problem, then a line with just '.' to search (q to quit):", fg="cyan"))
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            return "\n".join(lines) or None
        if not lines and line.strip().lower() in ("q", "quit", "exit"):
            return None
        if line.strip() == ".":
            return "\n".join(lines)
        lines.append(line)


@app.command()
def ask(
    query: str | None = typer.Argument(None, help="problem text; omit it to paste problems interactively"),
    file: str | None = typer.Option(None, "--file", help="read the problem text from a file"),
    config: str = "experiments/configs/submission.yaml",
    top_k: int = 5,
    lines: int = 12,
    set_: list[str] = typer.Option([], "--set", help="override, e.g. retrieval.dense_model=qwen3-emb-0.6b"),
) -> None:
    """Rank the 8,765 AppsRetrieval solutions for a programming problem with the submission pipeline."""
    import sys
    import time
    from pathlib import Path

    from seraph.config import load_config
    from seraph.memory import InMemoryChunkStore
    from seraph.retrieval.pipeline import Pipeline

    t0 = time.perf_counter()
    cfg = load_config(config, _parse_overrides(set_))
    data = load_apps(cfg.cache_dir)
    pipe = Pipeline.from_config(cfg, InMemoryChunkStore(data.chunks()))
    pipe.search("warm up", top_k=1)
    known = _known_answers(data)
    typer.echo(
        f"{cfg.name}: {len(data.corpus):,} solutions, dense model {cfg.retrieval.dense_model}, "
        f"ready in {time.perf_counter() - t0:.1f}s"
    )

    def answer(problem: str) -> None:
        problem = problem.strip()
        if not problem:
            return
        start = time.perf_counter()
        resp = pipe.search(problem, top_k=max(top_k, 100))
        ms = (time.perf_counter() - start) * 1000
        gold = known.get(" ".join(problem.split()), set())
        ranked = [r.chunk_hash for r in resp.results]
        title = " ".join(problem.split())[:90]
        typer.echo(typer.style(f"\n{title}{'…' if len(problem) > 90 else ''}", bold=True))
        note = f"searched {len(data.corpus):,} solutions in {ms:.0f} ms"
        if gold:
            hit = next((i for i, d in enumerate(ranked, 1) if d in gold), None)
            note += f" · known correct solution at #{hit}" if hit else " · known correct solution not in the top 100"
        typer.echo(typer.style(note, fg="cyan"))
        for i, r in enumerate(resp.results[:top_k], 1):
            views = "  ".join(f"{k} {v:.2f}" for k, v in r.retrieval_scores.items())
            mark = typer.style("  ✓ correct", fg="green", bold=True) if r.chunk_hash in gold else ""
            typer.echo(f"\n{typer.style(f'#{i}', bold=True)}  {r.chunk_hash}  score {r.score:.3f}{mark}")
            if views:
                typer.echo(typer.style(f"    {views}", dim=True))
            code = data.corpus[r.chunk_hash].strip().splitlines()
            for line in code[:lines]:
                typer.echo(f"    {line}")
            if len(code) > lines:
                typer.echo(typer.style(f"    … {len(code) - lines} more lines", dim=True))

    if file:
        answer(Path(file).read_text(encoding="utf-8"))
    elif query:
        answer(query)
    elif not sys.stdin.isatty():
        answer(sys.stdin.read())
    else:
        while (problem := _read_problem()) is not None:
            answer(problem)


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
            overrides = {**s.get("overrides", {}), "name": f"b2_dense_{model}", "retrieval": {"dense_model": model}}
            cfg = load_config(base, overrides)
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


@app.command("train-classifier")
def train_classifier(data: str, cache_dir: str = ".seraph_cache") -> None:
    """Train the learned query-type classifier from JSONL lines {"query": ..., "type": ...}."""
    from pathlib import Path

    from seraph.query.classifier import LearnedClassifier

    rows = [json.loads(line) for line in Path(data).read_text(encoding="utf-8").splitlines() if line.strip()]
    clf = LearnedClassifier().fit([r["query"] for r in rows], [r["type"] for r in rows])
    out = Path(cache_dir) / "query_classifier.pkl"
    clf.save(out)
    typer.echo(f"trained on {len(rows)} queries -> {out}")


@app.command("train-structural")
def train_structural(cache_dir: str = ".seraph_cache", out: str = "experiments/models/structural.pkl", c: float = 1.0) -> None:
    """Fit the query -> structural-trait model on the AppsRetrieval fit split."""
    from seraph.retrieval.structural import train_on_apps

    model = train_on_apps(cache_dir, out, c)
    trained = sum(m is not None for m in model.models)
    typer.echo(f"{len(model.vocab)} traits ({trained} predicted, rest prior-only) -> {out}")


@app.command("tune-structural")
def tune_structural(
    config: str = "experiments/configs/b3_hybrid_weighted.yaml",
    split: list[str] = typer.Option(["dev", "dev_stdin"], "--split"),
) -> None:
    """Retrieve views once, add the gamma view over the fused pool, grid-search its weight."""
    from seraph.config import load_config
    from seraph.evaluation.coir import score_run
    from seraph.memory import InMemoryChunkStore
    from seraph.retrieval.fusion import fuse
    from seraph.retrieval.pipeline import Pipeline
    from seraph.types import AnalyzedQuery

    cfg = load_config(config, {"retrieval": {"use_structural": True}})
    data = load_apps(cfg.cache_dir)
    pipe = Pipeline.from_config(cfg, InMemoryChunkStore(data.chunks()))
    base = dict(cfg.retrieval.static_weights)
    for name in split:
        s = data.split(name)
        qs = [AnalyzedQuery.plain(t, qid=q) for q, t in s.queries.items()]
        per_view = pipe.retrieve_views(qs, cfg.retrieval.first_stage_k)
        per_view["structural"] = pipe.structural.rescore(qs, pipe.fuse(qs, per_view))
        for g in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
            w = {k: v * (1 - g) for k, v in base.items() if k in per_view} | {"structural": g}
            run_ = {}
            for i, q in enumerate(qs):
                hits = fuse({v: h[i] for v, h in per_view.items()}, "weighted", w, norm=cfg.retrieval.fusion_norm)
                run_[q.qid] = {h.chunk_hash: h.score for h in hits[:100]}
            m = score_run(run_, s.qrels)
            typer.echo(f"{name:10s} gamma={g:<5} ndcg@10={m['ndcg@10']:.4f} mrr@10={m['mrr@10']:.4f} r@10={m['recall@10']:.4f}")


@app.command("tune-adaptive")
def tune_adaptive(
    config: str = "experiments/configs/b9_structural.yaml",
    split: str = "dev",
    folds: int = 5,
    step: float = 0.1,
    top_cells: list[int] = typer.Option([3, 6], "--top-cells"),
    out: str = "experiments/results/adaptive_weights.md",
    model_out: str = "experiments/models/adaptive_weights.pkl",
) -> None:
    """Static vs rule vs learned (k-fold CV) adaptive weights vs the per-query oracle, on dev."""
    from pathlib import Path

    import numpy as np
    from sklearn.model_selection import KFold

    from seraph.config import load_config
    from seraph.memory import InMemoryChunkStore
    from seraph.query.weights import LearnedWeights, RuleWeights, cell_utilities, query_features
    from seraph.retrieval.pipeline import Pipeline
    from seraph.types import AnalyzedQuery

    cfg = load_config(config)
    r = cfg.retrieval
    data = load_apps(cfg.cache_dir)
    s = data.split(split)
    stdin_ids = set(data.split("dev_stdin").queries) if split == "dev" else set()
    pipe = Pipeline.from_config(cfg, InMemoryChunkStore(data.chunks()))
    qs = [AnalyzedQuery.plain(t, qid=q) for q, t in s.queries.items()]
    per_view = pipe.retrieve_views(qs, r.first_stage_k)
    if pipe.structural is not None:
        per_view["structural"] = pipe.structural.rescore(qs, pipe.fuse(qs, per_view))
    views = list(per_view)
    runs = [{v: per_view[v][i] for v in views} for i in range(len(qs))]
    rels = [s.qrels.get(q.qid, {}) for q in qs]
    model = LearnedWeights(views, step, norm=r.fusion_norm)
    grid = model.grid
    util = cell_utilities(runs, rels, views, grid, r.fusion_norm)
    x = np.stack([query_features(q, run_, views) for q, run_ in zip(qs, runs, strict=True)])

    def util_of(weights: list[dict[str, float]]) -> np.ndarray:
        cells = np.array([[w.get(v, 0.0) for v in views] for w in weights])
        return np.array([
            cell_utilities([run_], [rel], views, c[None], r.fusion_norm)[0, 0]
            for run_, rel, c in zip(runs, rels, cells, strict=True)
        ])

    static = util_of([dict(r.static_weights)] * len(qs))
    rules = util_of(RuleWeights().weights_batch(qs, per_view))
    cv_static = np.zeros(len(qs))
    variants = [(kind, n) for kind in ("ridge", "lgbm") for n in (0, *top_cells)]
    learned = {v: np.zeros(len(qs)) for v in variants}
    for tr, te in KFold(folds, shuffle=True, random_state=cfg.seed).split(x):
        best = util[tr].mean(axis=0).argmax()
        cv_static[te] = util[te, best]
        for (kind, n), arr in learned.items():
            m = LearnedWeights(views, step, kind, r.fusion_norm, n).fit(x[tr], util[tr])
            arr[te] = util[te, m.predict_cells(x[te])]
    best_variant = max(learned, key=lambda v: learned[v].mean())
    oracle = util.max(axis=1)
    best_cell = grid[util.mean(axis=0).argmax()]
    rows = [
        (f"static tuned ({', '.join(f'{v}={r.static_weights.get(v, 0):g}' for v in views)})", static),
        (f"best grid cell, in-sample ({', '.join(f'{v}={w:.1f}' for v, w in zip(views, best_cell, strict=True))})",
         util[:, util.mean(axis=0).argmax()]),
        (f"best grid cell, {folds}-fold CV", cv_static),
        ("rule-based adaptive", rules),
        *[
            (f"learned adaptive, {'LightGBM' if k == 'lgbm' else k}, {f'top {n} cells' if n else 'all cells'} "
             f"({folds}-fold CV)", u)
            for (k, n), u in learned.items()
        ],
        ("per-query oracle (upper bound)", oracle),
    ]
    mask = np.array([q.qid in stdin_ids for q in qs])
    head = f"| weighting | {split} NDCG@10 |" + (" dev_stdin NDCG@10 |" if mask.any() else "")
    lines = [
        f"Views: {', '.join(views)}; grid step {step} ({len(grid)} cells); {len(qs)} {split} queries; "
        f"norm={r.fusion_norm}. NDCG@10 computed from cached per-view runs.",
        "",
        head,
        "|---|---|" + ("---|" if mask.any() else ""),
    ]
    for name, u in rows:
        line = f"| {name} | {u.mean():.4f} |"
        if mask.any():
            line += f" {u[mask].mean():.4f} |"
        lines.append(line)
    md = "\n".join(lines) + "\n"
    typer.echo(md)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(md, encoding="utf-8")
    kind, n = best_variant
    LearnedWeights(views, step, kind, r.fusion_norm, n).fit(x, util).save(model_out)
    typer.echo(f"saved best CV variant ({kind}, top_cells={n}) trained on all {split} queries -> {model_out}")


@app.command()
def ablate(
    spec: str = "experiments/configs/ablation.yaml",
    final: bool = typer.Option(False, "--final", help="Run every rung once on the AppsRetrieval test split"),
    skip_api: bool = typer.Option(False, "--skip-api", help="Skip rungs that call paid/quota-limited APIs"),
    force: bool = typer.Option(False, "--force", help="Rerun even when a result with the same config hash exists"),
    out: str | None = None,
) -> None:
    """Run the ablation ladder and write a markdown table."""
    from pathlib import Path

    from seraph.evaluation.ablations import Ablation, render, run_ablation

    ab = Ablation.load(spec)
    md = render(ab, run_ablation(ab, final, skip_api, force), final)
    typer.echo(md)
    path = Path(out or f"experiments/results/ablation{'_test' if final else ''}.md")
    path.write_text(md, encoding="utf-8")
    typer.echo(f"-> {path}")


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
