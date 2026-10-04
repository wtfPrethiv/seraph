"""Command-line interface for the versioned index."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .index import VersionedIndex


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
    commands.add_parser("versions", help="List indexed commit IDs")
    args = parser.parse_args()
    with VersionedIndex(args.repo, args.db) as store:
        if args.command == "index":
            print(json.dumps(asdict(store.index_commit(args.ref, args.base)), indent=2))
        elif args.command == "search":
            print(json.dumps([{"score": hit.score, **asdict(hit.chunk)} for hit in
                              store.search(args.query, args.ref, args.limit, args.history)], indent=2))
        else:
            print(json.dumps(store.indexed_versions(), indent=2))


if __name__ == "__main__":
    main()
