"""Small stdio MCP facade for Morpheus and other coding agents."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .index import VersionedIndex
from .service import search_index

mcp = FastMCP("Seraph")


def _repo() -> Path:
    repo = os.environ.get("SERAPH_REPO")
    if repo:
        return Path(repo)
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode != 0:
        raise ValueError("not inside a Git repository; set SERAPH_REPO to the repository to search")
    return Path(top.stdout.strip())


def _store() -> VersionedIndex:
    db = os.environ.get("SERAPH_DB")
    return VersionedIndex(_repo(), Path(db) if db else None)


def _results(query: str, version: str, limit: int, include_history: bool) -> dict:
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    with _store() as store:
        out = search_index(store, query, version, limit, include_history)
    for r in out["results"]:
        r["text"] = r["text"][:4000]
    return out


@mcp.tool()
def search_code(query: str, limit: int = 5) -> dict:
    """Find code by meaning in the current commit of the repository; use before grep when the exact name is unknown."""
    return _results(query, "HEAD", limit, False)


@mcp.tool()
def search_at_version(query: str, version: str, limit: int = 5) -> dict:
    """Find relevant code as it existed at a Git commit, branch, or tag."""
    return _results(query, version, limit, False)


@mcp.tool()
def search_history(query: str, limit: int = 5) -> dict:
    """Find relevant code across every indexed version, collapsing near-identical copies of the same symbol."""
    return _results(query, "HEAD", limit, True)


@mcp.tool()
def index_repository(version: str = "HEAD") -> dict:
    """Index one Git version and report changed-file work and reused chunks."""
    with _store() as store:
        start = time.perf_counter()
        stats = asdict(store.index_commit(version))
    return {**stats, "ms": round((time.perf_counter() - start) * 1000, 1)}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
