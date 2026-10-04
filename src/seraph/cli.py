"""Command-line interface for the versioned index."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .index import VersionedIndex


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
    commands.add_parser("versions", help="List indexed commit IDs")
    args = parser.parse_args()
    with VersionedIndex(args.repo, args.db) as store:
        if args.command == "index":
            print(json.dumps(asdict(store.index_commit(args.ref, args.base)), indent=2))
        elif args.command == "search" and args.engine == "baseline":
            print(json.dumps([{"score": hit.score, **asdict(hit.chunk)} for hit in
                              store.search(args.query, args.ref, args.limit, args.history)], indent=2))
        elif args.command == "search":
            from .service import search_index

            out = search_index(store, args.query, args.ref, args.limit, args.history)
            print(json.dumps(out, indent=2) if args.json else _render(out))
        else:
            print(json.dumps(store.indexed_versions(), indent=2))


if __name__ == "__main__":
    main()
