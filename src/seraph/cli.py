"""Command-line interface for the versioned index."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .index import VersionedIndex
from .service import compare_versions, find_dependencies, find_symbol, search_index


def _render(out: dict) -> str:
    idx = out["index"]
    lines = [
        f"commit {out['resolved_commit'][:10]}  indexed in {idx['ms']:.0f} ms "
        f"(parsed {idx['parsed_files']} files, reused {idx['reused_chunks']}/{idx['total_chunks']} chunks)  "
        f"searched in {out.get('search_ms', 0):.0f} ms",
        "",
    ]
    for rank, r in enumerate(out["results"], 1):
        where = f"{r['path']}:{r['start_line']}-{r['end_line']}"
        lines.append(f"{rank:>2}. {r['score']:.3f}  {where}  {r['symbol'] or ''}  @{r['commit'][:7]}")
        body = r["text"].splitlines()
        lines += [f"      {line}" for line in body[:8]]
        if len(body) > 8:
            lines.append(f"      ... {len(body) - 8} more lines")
        lines.append("")
    if not out["results"]:
        lines.append("no results")
    return "\n".join(lines)


def _render_symbol(out: dict) -> str:
    lines = [f"commit {out['resolved_commit'][:10]}", ""]
    for rank, r in enumerate(out["results"], 1):
        lines.append(f"{rank:>2}. {r['id']}  {r['path']}:{r['start_line']}-{r['end_line']}")
        body = r["text"].splitlines()
        lines += [f"      {line}" for line in body[:8]]
        if len(body) > 8:
            lines.append(f"      ... {len(body) - 8} more lines")
        lines.append("")
    if not out["results"]:
        lines.append("no matching symbol")
    return "\n".join(lines)


def _render_deps(out: dict) -> str:
    lines = [f"commit {out['resolved_commit'][:10]}  {out['direction']}", ""]
    if not out["results"]:
        return "\n".join(lines + ["no matching symbol"])
    for group in out["results"]:
        lines.append(group["symbol"])
        if not group["chains"]:
            lines.append("    no dependencies")
            continue
        for rank, chain in enumerate(group["chains"], 1):
            path = " -> ".join(chain["symbols"])
            kinds = ", ".join(e["kind"] for e in chain["edges"])
            lines.append(f"{rank:>3}. {chain['score']:.3f}  {path}")
            lines.append(f"      {kinds}")
        lines.append("")
    return "\n".join(lines)


def _render_compare(out: dict) -> str:
    s = out["summary"]
    lines = [
        f"{out['from'][:10]} -> {out['to'][:10]}",
        f"  added {s['added']}  modified {s['modified']}  renamed {s['renamed']}  "
        f"moved {s['moved']}  deleted {s['deleted']}",
        "",
        "symbols:",
    ]
    for e in out["symbols"]:
        extra = f"  (was {e['previous']})" if e.get("previous") else ""
        lines.append(f"  {e['change_type']:9} {e['symbol']}{extra}")
    lines += ["", "dependencies added:"]
    if out["dependencies"]["added"]:
        lines += [f"  {e['src']} --{e['kind']}--> {e['dst']}" for e in out["dependencies"]["added"]]
    else:
        lines.append("  none")
    lines += ["", "dependencies removed:"]
    if out["dependencies"]["removed"]:
        lines += [f"  {e['src']} --{e['kind']}--> {e['dst']}" for e in out["dependencies"]["removed"]]
    else:
        lines.append("  none")
    lines += ["", "commits:"]
    if out["commits"]:
        for c in out["commits"]:
            files = ", ".join(c["files"]) or "-"
            lines.append(f"  {c['commit'][:7]}  {c['subject']}  ({files})")
    else:
        lines.append("  none")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="seraph")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Git repository to index/search")
    parser.add_argument("--db", type=Path, help="SQLite index path (default: REPO/.seraph/index.sqlite)")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Index a Git commit")
    index.add_argument("--ref", default="HEAD")
    index.add_argument("--base", help="Previously indexed version to reuse")
    search = commands.add_parser("search", help="Search an indexed version")
    search.add_argument("query")
    search.add_argument("--ref", default="HEAD")
    search.add_argument("--history", action="store_true", help="Search all indexed versions")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--json", action="store_true", help="Print raw JSON results")
    search.add_argument("--engine", choices=("pipeline", "baseline"), default="pipeline",
                        help="retrieval pipeline (SERAPH_CONFIG) or the index's built-in lexical baseline")
    symbol = commands.add_parser("symbol", help="Look up a symbol by name or path::symbol")
    symbol.add_argument("name")
    symbol.add_argument("--ref", default="HEAD")
    symbol.add_argument("--json", action="store_true")
    deps = commands.add_parser("deps", help="Show dependency chains from a symbol")
    deps.add_argument("symbol")
    deps.add_argument("--ref", default="HEAD")
    deps.add_argument("--direction", choices=("out", "in", "both"), default="out")
    deps.add_argument("--depth", type=int, default=3)
    deps.add_argument("--limit", type=int, default=50)
    deps.add_argument("--json", action="store_true")
    compare = commands.add_parser("compare", help="Diff symbols, dependencies and commits between two versions")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.add_argument("--limit", type=int, default=200)
    compare.add_argument("--json", action="store_true")
    commands.add_parser("versions", help="List indexed commit IDs")
    args = parser.parse_args()
    with VersionedIndex(args.repo, args.db) as store:
        if args.command == "index":
            print(json.dumps(asdict(store.index_commit(args.ref, args.base)), indent=2))
        elif args.command == "search" and args.engine == "baseline":
            print(json.dumps([{"score": hit.score, **asdict(hit.chunk)} for hit in
                              store.search(args.query, args.ref, args.limit, args.history)], indent=2))
        elif args.command == "search":
            out = search_index(store, args.query, args.ref, args.limit, args.history)
            print(json.dumps(out, indent=2) if args.json else _render(out))
        elif args.command == "symbol":
            out = find_symbol(store, args.name, args.ref)
            print(json.dumps(out, indent=2) if args.json else _render_symbol(out))
        elif args.command == "deps":
            out = find_dependencies(store, args.symbol, args.ref, args.direction, max_depth=args.depth,
                                    limit=args.limit)
            print(json.dumps(out, indent=2) if args.json else _render_deps(out))
        elif args.command == "compare":
            out = compare_versions(store, args.a, args.b, args.limit)
            print(json.dumps(out, indent=2) if args.json else _render_compare(out))
        else:
            print(json.dumps(store.indexed_versions(), indent=2))


if __name__ == "__main__":
    main()
