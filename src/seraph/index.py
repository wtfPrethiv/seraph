"""A small Git-aware chunk store and lexical search baseline.

The retrieval teammate can use ``iter_chunks`` to build an embedding index and
``search`` as a fallback. Content hashes cache text; occurrence IDs distinguish
identical text appearing in different files or commits.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import Iterator


SUPPORTED_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go"}
MAX_FILE_BYTES = 1_000_000
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")


@dataclass(frozen=True)
class Chunk:
    occurrence_id: str
    commit: str
    path: str
    symbol: str | None
    kind: str
    start_line: int
    end_line: int
    content_hash: str
    text: str
    language: str


@dataclass(frozen=True)
class IndexStats:
    commit: str
    base_commit: str | None
    parsed_files: int
    reused_chunks: int
    total_chunks: int


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tokens(value: str) -> list[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return [token.lower() for token in TOKEN_RE.findall(value.replace("_", " "))]


def _language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {".py": "python", ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript", ".go": "go"}[suffix]


def _python_chunks(text: str) -> list[tuple[str | None, str, int, int, str]]:
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [(None, "file", 1, max(len(lines), 1), text)]

    chunks: list[tuple[str | None, str, int, int, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min((decorator.lineno for decorator in node.decorator_list), default=node.lineno)
            end = node.end_lineno or node.lineno
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append((node.name, kind, start, end, "".join(lines[start - 1:end])))
    if not chunks:
        chunks.append((None, "file", 1, max(len(lines), 1), text))
    return chunks


def _chunks(path: str, text: str) -> list[tuple[str | None, str, int, int, str]]:
    if Path(path).suffix.lower() == ".py":
        return _python_chunks(text)
    # File-level fallback keeps other languages searchable until their parsers land.
    return [(None, "file", 1, max(len(text.splitlines()), 1), text)]


class VersionedIndex:
    def __init__(self, repo: str | Path, db_path: str | Path | None = None):
        self.repo = Path(repo).resolve()
        if not (self.repo / ".git").exists():
            raise ValueError(f"Not a Git checkout: {self.repo}")
        self.db_path = Path(db_path) if db_path else self.repo / ".seraph" / "index.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        stored_repo = self.conn.execute("SELECT value FROM metadata WHERE key='repo'").fetchone()
        if stored_repo and stored_repo[0] != str(self.repo):
            raise ValueError(f"Index belongs to another repository: {stored_repo[0]}")
        self.conn.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES ('repo', ?)", (str(self.repo),))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "VersionedIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS versions (
                commit_id TEXT PRIMARY KEY,
                base_commit TEXT,
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS blobs (
                content_hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                language TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS occurrences (
                occurrence_id TEXT PRIMARY KEY,
                commit_id TEXT NOT NULL REFERENCES versions(commit_id),
                path TEXT NOT NULL,
                symbol TEXT,
                kind TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content_hash TEXT NOT NULL REFERENCES blobs(content_hash)
            );
            CREATE INDEX IF NOT EXISTS occurrences_commit ON occurrences(commit_id);
            CREATE INDEX IF NOT EXISTS occurrences_path ON occurrences(commit_id, path);
            CREATE INDEX IF NOT EXISTS occurrences_hash ON occurrences(content_hash);
        """)

    def _git(self, *args: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise ValueError(result.stderr.decode("utf-8", "replace").strip())
        return result.stdout

    def resolve(self, ref: str) -> str:
        return self._git("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()

    def indexed_versions(self) -> list[str]:
        rows = self.conn.execute("SELECT commit_id FROM versions ORDER BY indexed_at").fetchall()
        return [row[0] for row in rows]

    def index_commit(self, ref: str = "HEAD", base_ref: str | None = None) -> IndexStats:
        commit = self.resolve(ref)
        existing = self.conn.execute("SELECT base_commit FROM versions WHERE commit_id=?", (commit,)).fetchone()
        if existing:
            count = self.conn.execute("SELECT COUNT(*) FROM occurrences WHERE commit_id=?", (commit,)).fetchone()[0]
            return IndexStats(commit, existing[0], 0, count, count)

        base: str | None = None
        if base_ref:
            candidate = self.resolve(base_ref)
            if self.conn.execute("SELECT 1 FROM versions WHERE commit_id=?", (candidate,)).fetchone():
                base = candidate
        else:
            parents = self._git("rev-list", "--parents", "-n", "1", commit).decode().split()
            if len(parents) > 1 and self.conn.execute(
                "SELECT 1 FROM versions WHERE commit_id=?", (parents[1],)
            ).fetchone():
                base = parents[1]

        paths = self._git("ls-tree", "-r", "--name-only", commit).decode("utf-8", "replace").splitlines()
        eligible = {path for path in paths if Path(path).suffix.lower() in SUPPORTED_SUFFIXES}
        changed = eligible
        reused = 0
        if base:
            diff = self._git("diff", "--name-only", base, commit).decode("utf-8", "replace").splitlines()
            changed = eligible.intersection(diff)

        parsed_files = 0
        with self.conn:
            self.conn.execute(
                "INSERT INTO versions(commit_id, base_commit, indexed_at) VALUES (?, ?, ?)",
                (commit, base, datetime.now(timezone.utc).isoformat()),
            )
            if base:
                rows = self.conn.execute(
                    "SELECT path, symbol, kind, start_line, end_line, content_hash "
                    "FROM occurrences WHERE commit_id=?", (base,)
                ).fetchall()
                for row in rows:
                    if row["path"] not in eligible or row["path"] in changed:
                        continue
                    occurrence_id = _digest("\0".join((commit, row["path"], row["symbol"] or "",
                                                       str(row["start_line"]), row["content_hash"])))
                    self.conn.execute(
                        "INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (occurrence_id, commit, row["path"], row["symbol"], row["kind"],
                         row["start_line"], row["end_line"], row["content_hash"]),
                    )
                    reused += 1

            for path in sorted(changed):
                raw = self._git("show", f"{commit}:{path}")
                if len(raw) > MAX_FILE_BYTES or b"\0" in raw:
                    continue
                text = raw.decode("utf-8", "replace")
                language = _language(path)
                parsed_files += 1
                for symbol, kind, start, end, part in _chunks(path, text):
                    # The same source text can mean different things in different languages.
                    content_hash = _digest(language + "\0" + part)
                    occurrence_id = _digest("\0".join((commit, path, symbol or "", str(start), content_hash)))
                    self.conn.execute(
                        "INSERT OR IGNORE INTO blobs VALUES (?, ?, ?)",
                        (content_hash, part, language),
                    )
                    self.conn.execute(
                        "INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (occurrence_id, commit, path, symbol, kind, start, end, content_hash),
                    )

        total = self.conn.execute("SELECT COUNT(*) FROM occurrences WHERE commit_id=?", (commit,)).fetchone()[0]
        return IndexStats(commit, base, parsed_files, reused, total)

    def iter_chunks(self, ref: str = "HEAD", include_history: bool = False) -> Iterator[Chunk]:
        commit = self.resolve(ref)
        if not self.conn.execute("SELECT 1 FROM versions WHERE commit_id=?", (commit,)).fetchone():
            raise ValueError(f"Version is not indexed: {ref} ({commit[:12]})")
        where = "" if include_history else "WHERE o.commit_id = ?"
        params = () if include_history else (commit,)
        rows = self.conn.execute(
            "SELECT o.*, b.text, b.language FROM occurrences o "
            f"JOIN blobs b ON b.content_hash = o.content_hash {where} "
            "ORDER BY o.commit_id, o.path, o.start_line", params
        )
        for row in rows:
            yield Chunk(row["occurrence_id"], row["commit_id"], row["path"], row["symbol"],
                        row["kind"], row["start_line"], row["end_line"],
                        row["content_hash"], row["text"], row["language"])

    def search(self, query: str, ref: str = "HEAD", limit: int = 10,
               include_history: bool = False) -> list[SearchHit]:
        """A deterministic lexical fallback; the model owner can replace ranking."""
        query_terms = _tokens(query)
        if not query_terms or limit < 1:
            return []
        chunks = list(self.iter_chunks(ref, include_history))
        if not chunks:
            return []
        documents = [Counter(_tokens(" ".join((chunk.path, chunk.symbol or "", chunk.text))))
                     for chunk in chunks]
        lengths = [sum(doc.values()) for doc in documents]
        avg_length = sum(lengths) / len(lengths) or 1.0
        document_frequency = Counter(term for doc in documents for term in doc)
        hits: list[SearchHit] = []
        for chunk, terms, length in zip(chunks, documents, lengths):
            score = 0.0
            for term in set(query_terms):
                frequency = terms[term]
                if not frequency:
                    continue
                idf = math.log(1 + (len(chunks) - document_frequency[term] + 0.5)
                               / (document_frequency[term] + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / avg_length))
            if score > 0:
                hits.append(SearchHit(chunk, score))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.path, hit.chunk.start_line))
        return hits[:limit]
