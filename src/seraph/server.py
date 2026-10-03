"""Small stdio MCP facade for Morpheus and other coding agents."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .index import VersionedIndex


mcp = FastMCP("Seraph")


def _store() -> VersionedIndex:
    repo = os.environ.get("SERAPH_REPO")
    if not repo:
        raise ValueError("SERAPH_REPO must point to the repository to search")
    db = os.environ.get("SERAPH_DB")
    return VersionedIndex(Path(repo), Path(db) if db else None)


def _results(query: str, version: str, limit: int, include_history: bool) -> dict:
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    with _store() as store:
        stats = store.index_commit(version)
        hits = store.search(query, version, limit, include_history)
        return {
            "query": query,
            "requested_version": version,
            "resolved_commit": stats.commit,
            "results": [
                {"score": round(hit.score, 4), **asdict(hit.chunk),
                 "text": hit.chunk.text[:4000]}
                for hit in hits
            ],
        }


@mcp.tool()
def search_code(query: str, limit: int = 5) -> dict:
    """Find relevant code in the current commit of the configured repository."""
    return _results(query, "HEAD", limit, False)


@mcp.tool()
def search_at_version(query: str, version: str, limit: int = 5) -> dict:
    """Find relevant code as it existed at a Git commit, branch, or tag."""
    return _results(query, version, limit, False)


@mcp.tool()
def index_repository(version: str = "HEAD") -> dict:
    """Index one Git version and report changed-file work and reused chunks."""
    with _store() as store:
        return asdict(store.index_commit(version))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
