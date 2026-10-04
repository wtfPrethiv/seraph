"""YAML-driven ablation ladder: run each rung, then render one markdown table."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from seraph.config import load_config
from seraph.evaluation.experiment import RESULTS_DIR, config_hash, run_experiment

log = logging.getLogger(__name__)


@dataclass
class Rung:
    label: str
    config: Path
    splits: list[str]
    api: bool = False
    base: str | None = None  # label the delta is computed against; default: previous rung


@dataclass
class Ablation:
    rungs: list[Rung]
    splits: list[str]
    not_applicable: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> Ablation:
        path = Path(path)
        spec = yaml.safe_load(path.read_text())
        splits = list(spec.get("splits", ["dev"]))
        rungs, prev = [], None
        for r in spec["ladder"]:
            rungs.append(
                Rung(r["label"], path.parent / r["config"], list(r.get("splits", splits)),
                     bool(r.get("api", False)), r.get("base", prev))
            )
            prev = r["label"]
        return cls(rungs, splits, list(spec.get("not_applicable", [])))


def _cached(name: str, split: str, chash: str, results_dir: Path) -> dict[str, Any] | None:
    path = results_dir / f"{name}__{split}.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    return res if res.get("meta", {}).get("config_hash") == chash else None


def run_ablation(
    ablation: Ablation,
    final: bool = False,
    skip_api: bool = False,
    force: bool = False,
    results_dir: Path = RESULTS_DIR,
) -> dict[tuple[str, str], dict[str, Any] | str]:
    """Returns {(label, split): result dict, or a short reason string when skipped/failed}."""
    from seraph.backends.common import MissingAPIKeyError, QuotaExhaustedError

    out: dict[tuple[str, str], dict[str, Any] | str] = {}
    for rung in ablation.rungs:
        cfg = load_config(rung.config)
        splits = ["test"] if final else rung.splits
        for split in splits:
            key = (rung.label, split)
            if rung.api and skip_api:
                out[key] = "skipped (API)"
                continue
            if not force and not final and (res := _cached(cfg.name, split, config_hash(cfg), results_dir)):
                out[key] = res
                continue
            try:
                out[key] = run_experiment(cfg, split=split, final=final, results_dir=results_dir)
            except (QuotaExhaustedError, MissingAPIKeyError) as e:
                log.warning("%s on %s: %s", rung.label, split, e)
                out[key] = "skipped (quota)" if isinstance(e, QuotaExhaustedError) else "skipped (no key)"
    return out


def render(ablation: Ablation, results: dict[tuple[str, str], dict[str, Any] | str], final: bool = False) -> str:
    splits = ["test"] if final else ablation.splits
    head = "| component |" + "".join(f" {s} NDCG@10 | {s} R@100 |" for s in splits) + " p50 latency |"
    lines = [head, "|---|" + "---|---|" * len(splits) + "---|"]
    def ndcg_of(label: str | None, split: str) -> float | None:
        res = results.get((label, split)) if label else None
        return res["metrics"]["ndcg@10"] if isinstance(res, dict) else None

    for rung in ablation.rungs:
        cells, latency = [], ""
        for split in splits:
            res = results.get((rung.label, split))
            if isinstance(res, dict):
                m = res["metrics"]
                ndcg = m["ndcg@10"]
                base = ndcg_of(rung.base, split)
                delta = f" ({ndcg - base:+.4f})" if base is not None else ""
                cells.append(f" {ndcg:.4f}{delta} | {m['recall@100']:.4f} |")
                latency = latency or f"{m.get('latency_p50_ms', 0):.1f} ms"
            else:
                cells.append(f" {res or 'n/a'} | |")
        lines.append(f"| {rung.label} |" + "".join(cells) + f" {latency} |")
    for na in ablation.not_applicable:
        lines.append(f"| {na['label']} |" + " n/a | |" * len(splits) + f" {na.get('reason', '')} |")
    return "\n".join(lines) + "\n"
