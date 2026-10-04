"""A small Git-aware chunk store and lexical search baseline.

The retrieval teammate can use ``iter_chunks`` to build an embedding index and
``search`` as a fallback. Content hashes cache text; occurrence IDs distinguish
identical text appearing in different files or commits.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
import sqlite3
import subprocess
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

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


_TS_DECLS = {
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
    "lexical_declaration": "variable",
    "variable_declaration": "variable",
}
_TS_METHODS = {"method_definition", "public_field_definition", "field_definition"}
_GO_DECLS = {"function_declaration": "function", "method_declaration": "method", "type_declaration": "type"}


@cache
def _parser(language: str):
    from tree_sitter import Language, Parser

    if language == "go":
        import tree_sitter_go as grammar

        return Parser(Language(grammar.language()))
    if language == "javascript":
        import tree_sitter_javascript as grammar

        return Parser(Language(grammar.language()))
    import tree_sitter_typescript as grammar

    tsx = language == "tsx"
    return Parser(Language(grammar.language_tsx() if tsx else grammar.language_typescript()))


def _node_name(node) -> str | None:
    name = node.child_by_field_name("name")
    if name is None and node.type in ("lexical_declaration", "variable_declaration", "type_declaration"):
        for child in node.named_children:
            name = child.child_by_field_name("name")
            if name is not None:
                break
    if name is None and node.type == "method_declaration":
        name = node.child_by_field_name("name")
    return name.text.decode("utf8", "replace") if name is not None else None


def _go_receiver(node) -> str | None:
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return None
    names = [n.text.decode() for n in _walk(receiver) if n.type == "type_identifier"]
    return names[0] if names else None


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _tree_chunks(path: str, text: str) -> list[tuple[str | None, str, int, int, str]]:
    suffix = Path(path).suffix.lower()
    language = {".go": "go", ".js": "javascript", ".jsx": "javascript", ".tsx": "tsx"}.get(suffix, "typescript")
    lines = text.splitlines(keepends=True)
    try:
        root = _parser(language).parse(text.encode("utf8")).root_node
    except Exception:
        return [(None, "file", 1, max(len(lines), 1), text)]

    def emit(node, symbol: str | None, kind: str, start_node=None) -> None:
        start = (start_node or node).start_point[0] + 1
        end = node.end_point[0] + 1
        chunks.append((symbol, kind, start, end, "".join(lines[start - 1:end])))

    chunks: list[tuple[str | None, str, int, int, str]] = []
    decls = _GO_DECLS if language == "go" else _TS_DECLS
    for top in root.named_children:
        node = top
        if node.type == "export_statement":
            inner = node.child_by_field_name("declaration") or next(
                (c for c in node.named_children if c.type in decls), None
            )
            if inner is None:
                continue
            node = inner
        kind = decls.get(node.type)
        if kind is None or node.end_point[0] == node.start_point[0]:
            continue
        name = _node_name(node)
        if language == "go" and kind == "method":
            receiver = _go_receiver(node)
            name = f"{receiver}.{name}" if receiver and name else name
        body = node.child_by_field_name("body")
        methods = [c for c in body.named_children if c.type in _TS_METHODS] if kind == "class" and body else []
        multi_line = [m for m in methods if m.end_point[0] > m.start_point[0]]
        if multi_line:
            first = multi_line[0]
            header_end = first.start_point[0]
            start = top.start_point[0] + 1
            if header_end >= start:
                chunks.append((name, "class", start, header_end, "".join(lines[start - 1:header_end])))
            for method in multi_line:
                method_name = _node_name(method)
                emit(method, f"{name}.{method_name}" if name and method_name else method_name, "method")
        else:
            emit(node, name, kind, start_node=top)
    if not chunks:
        chunks.append((None, "file", 1, max(len(lines), 1), text))
    return chunks


def _chunks(path: str, text: str) -> list[tuple[str | None, str, int, int, str]]:
    if Path(path).suffix.lower() == ".py":
        return _python_chunks(text)
    return _tree_chunks(path, text)


class VersionedIndex:
    def __init__(self, repo: str | Path, db_path: str | Path | None = None):
        self.repo = Path(repo).resolve()
        if not (self.repo / ".git").exists():
            raise ValueError(f"Not a Git checkout: {self.repo}")
        self.db_path = Path(db_path) if db_path else self.repo / ".seraph" / "index.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not db_path:
            ignore = self.db_path.parent / ".gitignore"
            if not ignore.exists():
                ignore.write_text("*\n")
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

    def __enter__(self) -> VersionedIndex:
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
            CREATE TABLE IF NOT EXISTS refs (
                commit_id TEXT NOT NULL REFERENCES versions(commit_id),
                path TEXT NOT NULL,
                symbol TEXT,
                kind TEXT NOT NULL,
                target TEXT NOT NULL,
                local TEXT
            );
            CREATE INDEX IF NOT EXISTS refs_commit ON refs(commit_id, path);
            -- Per-commit data added after a commit may first have been indexed (backfilled on demand).
            CREATE TABLE IF NOT EXISTS derived (
                commit_id TEXT NOT NULL REFERENCES versions(commit_id),
                feature TEXT NOT NULL,
                PRIMARY KEY (commit_id, feature)
            );
            -- One row per symbol occurrence (or deletion) per commit; lineage_id survives renames and moves.
            CREATE TABLE IF NOT EXISTS lineage (
                commit_id TEXT NOT NULL REFERENCES versions(commit_id),
                lineage_id TEXT NOT NULL,
                occurrence_id TEXT,
                path TEXT NOT NULL,
                symbol TEXT,
                change_type TEXT NOT NULL,
                previous TEXT,
                similarity REAL
            );
            CREATE INDEX IF NOT EXISTS lineage_by_id ON lineage(lineage_id);
            CREATE INDEX IF NOT EXISTS lineage_by_commit ON lineage(commit_id);
            CREATE TABLE IF NOT EXISTS lineage_base (
                commit_id TEXT PRIMARY KEY REFERENCES versions(commit_id),
                base TEXT
            );
        """)

    def _git(self, *args: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            capture_output=True,
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
        if base:
            self._ensure_refs(base)
        with self.conn:
            self.conn.execute(
                "INSERT INTO versions(commit_id, base_commit, indexed_at) VALUES (?, ?, ?)",
                (commit, base, datetime.now(UTC).isoformat()),
            )
            if base:
                for row in self.conn.execute("SELECT * FROM refs WHERE commit_id=?", (base,)).fetchall():
                    if row["path"] in eligible and row["path"] not in changed:
                        self.conn.execute("INSERT INTO refs VALUES (?, ?, ?, ?, ?, ?)",
                                          (commit, *tuple(row)[1:]))
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
                spans = _chunks(path, text)
                self._insert_refs(commit, path, text, spans)
                for symbol, kind, start, end, part in spans:
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
            self.conn.execute("INSERT INTO derived VALUES (?, 'refs')", (commit,))
        self._update_lineage(commit)

        total = self.conn.execute("SELECT COUNT(*) FROM occurrences WHERE commit_id=?", (commit,)).fetchone()[0]
        return IndexStats(commit, base, parsed_files, reused, total)

    def _is_ancestor(self, ancestor: str, commit: str) -> bool:
        result = subprocess.run(["git", "-C", str(self.repo), "merge-base", "--is-ancestor", ancestor, commit],
                                capture_output=True, check=False)
        return result.returncode == 0

    def _lineage_base(self, commit: str) -> str | None:
        """The nearest indexed first-parent ancestor, which lineage is matched against."""
        indexed = set(self.indexed_versions()) - {commit}
        if not indexed:
            return None
        history = self._git("rev-list", "--first-parent", "--max-count=5000", commit).decode().split()
        return next((c for c in history[1:] if c in indexed), None)

    def _has_lineage(self, commit: str) -> bool:
        return bool(self.conn.execute("SELECT 1 FROM lineage_base WHERE commit_id=?", (commit,)).fetchone())

    def _record_lineage(self, commit: str, base: str | None) -> None:
        from seraph.lineage import match_symbols

        old = list(self.iter_chunks(base)) if base else []
        previous = dict(self.conn.execute(
            "SELECT occurrence_id, lineage_id FROM lineage WHERE commit_id=? AND occurrence_id IS NOT NULL", (base,)
        ).fetchall()) if base else {}
        rows = []
        for m in match_symbols(old, list(self.iter_chunks(commit))):
            c = m.new or m.old
            assert c is not None
            lineage_id = previous.get(m.old.occurrence_id) if m.old else None
            lineage_id = lineage_id or _digest("\0".join((c.commit, c.path, c.symbol or "", str(c.start_line))))[:16]
            rows.append((commit, lineage_id, m.new.occurrence_id if m.new else None, c.path, c.symbol,
                         m.change_type.value, m.old.occurrence_id if m.old else None, m.similarity))
        with self.conn:
            self.conn.execute("DELETE FROM lineage WHERE commit_id=?", (commit,))
            self.conn.executemany("INSERT INTO lineage VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            self.conn.execute("INSERT OR REPLACE INTO lineage_base VALUES (?, ?)", (commit, base))

    def ensure_lineage(self, commit: str) -> None:
        """Record lineage for `commit` and any indexed ancestors that lack it, oldest first."""
        pending: list[str] = []
        current: str | None = commit
        while current is not None and not self._has_lineage(current) and current not in pending:
            pending.append(current)
            current = self._lineage_base(current)
        for c in reversed(pending):
            self._record_lineage(c, self._lineage_base(c))

    def _update_lineage(self, commit: str) -> None:
        # An older commit indexed after its descendants changes their bases, so replay every commit in order.
        descendants = [c for c in self.indexed_versions() if c != commit and self._has_lineage(c)
                       and self._is_ancestor(commit, c)]
        if not descendants:
            self.ensure_lineage(commit)
            return
        for c in self.ordered_versions():
            self._record_lineage(c, self._lineage_base(c))

    def ordered_versions(self) -> list[str]:
        """Indexed commits, ancestors before descendants."""
        indexed = self.indexed_versions()
        if not indexed:
            return []
        order = self._git("rev-list", "--topo-order", "--reverse", *indexed).decode().split()
        position = {c: i for i, c in enumerate(order)}
        return sorted(indexed, key=lambda c: position.get(c, -1))

    def lineage_ids(self, commit: str) -> dict[str, str]:
        """Occurrence id -> lineage id for one indexed commit."""
        self.ensure_lineage(commit)
        return dict(self.conn.execute(
            "SELECT occurrence_id, lineage_id FROM lineage WHERE commit_id=? AND occurrence_id IS NOT NULL", (commit,)
        ).fetchall())

    def lineage_rows(self, lineage_id: str | None = None) -> list[sqlite3.Row]:
        if lineage_id is None:
            return self.conn.execute("SELECT * FROM lineage").fetchall()
        return self.conn.execute("SELECT * FROM lineage WHERE lineage_id=?", (lineage_id,)).fetchall()

    def lineage_bases(self) -> dict[str, str | None]:
        return dict(self.conn.execute("SELECT commit_id, base FROM lineage_base").fetchall())

    def get_chunk(self, occurrence_id: str) -> Chunk:
        row = self.conn.execute(
            "SELECT o.*, b.text, b.language FROM occurrences o JOIN blobs b ON b.content_hash = o.content_hash "
            "WHERE o.occurrence_id=?", (occurrence_id,)
        ).fetchone()
        if row is None:
            raise KeyError(occurrence_id)
        return Chunk(row["occurrence_id"], row["commit_id"], row["path"], row["symbol"], row["kind"],
                     row["start_line"], row["end_line"], row["content_hash"], row["text"], row["language"])

    def find_lineage_pair(self, symbol: str, a: str, b: str) -> tuple[Chunk | None, Chunk | None]:
        """The occurrences of one symbol lineage in commits `a` and `b`; `symbol` is `path::symbol` or a name."""
        self.ensure_lineage(a)
        self.ensure_lineage(b)

        def matching(commit: str) -> list[sqlite3.Row]:
            rows = self.conn.execute(
                "SELECT * FROM lineage WHERE commit_id=? AND occurrence_id IS NOT NULL ORDER BY path", (commit,)
            ).fetchall()
            return [r for r in rows if r["symbol"] and symbol in (
                f"{r['path']}::{r['symbol']}", r["symbol"], r["symbol"].rsplit(".", 1)[-1])]

        hits = matching(b) or matching(a)
        if not hits:
            return None, None
        lineage_id = hits[0]["lineage_id"]

        def occurrence(commit: str) -> Chunk | None:
            row = self.conn.execute(
                "SELECT occurrence_id FROM lineage WHERE commit_id=? AND lineage_id=? AND occurrence_id IS NOT NULL",
                (commit, lineage_id),
            ).fetchone()
            return self.get_chunk(row[0]) if row else None

        return occurrence(a), occurrence(b)

    def commit_info(self, commits: list[str]) -> dict[str, tuple[int, str]]:
        """Commit -> (commit timestamp, subject line)."""
        if not commits:
            return {}
        out = {}
        for line in self._git("show", "-s", "--format=%H%x00%ct%x00%s", *commits).decode("utf-8", "replace").splitlines():
            parts = line.split("\0")
            if len(parts) == 3:
                out[parts[0]] = (int(parts[1]), parts[2])
        return out

    def tags(self) -> dict[str, str]:
        """Commit -> tag name (annotated tags are peeled to their commit)."""
        raw = self._git("for-each-ref", "--format=%(objectname)%00%(*objectname)%00%(refname:short)", "refs/tags")
        out = {}
        for line in raw.decode("utf-8", "replace").splitlines():
            obj, peeled, name = line.split("\0")
            out.setdefault(peeled or obj, name)
        return out

    def _insert_refs(self, commit: str, path: str, text: str, spans) -> None:
        from seraph.graph.builder import extract_refs

        self.conn.executemany(
            "INSERT INTO refs VALUES (?, ?, ?, ?, ?, ?)",
            [(commit, path, r.symbol, str(r.kind), r.target, r.local) for r in extract_refs(path, text, spans)],
        )

    def _has(self, commit: str, feature: str) -> bool:
        return bool(self.conn.execute(
            "SELECT 1 FROM derived WHERE commit_id=? AND feature=?", (commit, feature)
        ).fetchone())

    def _ensure_refs(self, commit: str) -> None:
        """Extract references for a commit indexed before the `refs` table existed."""
        if self._has(commit, "refs"):
            return
        paths = self.conn.execute("SELECT DISTINCT path FROM occurrences WHERE commit_id=?", (commit,)).fetchall()
        with self.conn:
            self.conn.execute("DELETE FROM refs WHERE commit_id=?", (commit,))
            for (path,) in paths:
                text = self._git("show", f"{commit}:{path}").decode("utf-8", "replace")
                self._insert_refs(commit, path, text, _chunks(path, text))
            self.conn.execute("INSERT INTO derived VALUES (?, 'refs')", (commit,))

    def iter_refs(self, ref: str = "HEAD"):
        """`(path, graph.builder.Ref)` for every reference extracted from an indexed commit."""
        from seraph.graph.builder import Ref

        commit = self.resolve(ref)
        if not self.conn.execute("SELECT 1 FROM versions WHERE commit_id=?", (commit,)).fetchone():
            raise ValueError(f"Version is not indexed: {ref} ({commit[:12]})")
        self._ensure_refs(commit)
        for row in self.conn.execute("SELECT * FROM refs WHERE commit_id=? ORDER BY rowid", (commit,)):
            yield row["path"], Ref(row["symbol"], row["kind"], row["target"], row["local"])

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
        for chunk, terms, length in zip(chunks, documents, lengths, strict=True):
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
