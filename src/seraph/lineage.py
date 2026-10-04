"""Symbol lineage across indexed commits: matching, rename/move detection, and a `VersionStore`.

A symbol keeps its lineage id while it is modified, renamed or moved. Matching runs in passes:
same path and name, then identical content elsewhere, then body similarity with the symbol's own
name masked out (token 3-gram Jaccard), so a renamed-and-edited function still links up.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from seraph.types import ChangeType, LineageNode, SymbolDiff, Version

if TYPE_CHECKING:
    from seraph.index import Chunk, VersionedIndex

SIMILARITY_THRESHOLD = 0.5
MIN_SHINGLES = 8  # tiny bodies look alike; they only link by identical content
_TOKEN = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class Match:
    old: Chunk | None
    new: Chunk | None
    change_type: ChangeType
    similarity: float = 1.0


def _shingles(chunk: Chunk) -> set[tuple[str, ...]]:
    text = chunk.text
    if chunk.symbol:
        text = re.sub(rf"\b{re.escape(chunk.symbol.rsplit('.', 1)[-1])}\b", "\0", text)
    tokens = _TOKEN.findall(text)
    if len(tokens) < 3:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def _moved_or_renamed(old: Chunk, new: Chunk) -> ChangeType:
    return ChangeType.MOVED if old.symbol == new.symbol else ChangeType.RENAMED


def match_symbols(
    old: Sequence[Chunk], new: Sequence[Chunk], threshold: float = SIMILARITY_THRESHOLD
) -> list[Match]:
    """Pair the chunks of two versions; unpaired new chunks are added, unpaired old ones deleted."""
    matches: list[Match] = []

    def keyed(chunks: Sequence[Chunk]) -> dict[tuple, Chunk]:
        seen: dict[tuple, int] = defaultdict(int)
        out = {}
        for c in sorted(chunks, key=lambda c: (c.path, c.start_line)):
            key = (c.path, c.symbol)
            out[(*key, seen[key])] = c
            seen[key] += 1
        return out

    old_by_key, new_by_key = keyed(old), keyed(new)
    for key in old_by_key.keys() & new_by_key.keys():
        o, n = old_by_key.pop(key), new_by_key.pop(key)
        same = o.content_hash == n.content_hash
        matches.append(Match(o, n, ChangeType.UNCHANGED if same else ChangeType.MODIFIED))

    old_left, new_left = list(old_by_key.values()), list(new_by_key.values())
    by_hash: dict[str, list[Chunk]] = defaultdict(list)
    for o in old_left:
        by_hash[o.content_hash].append(o)
    still_new = []
    for n in new_left:
        candidates = by_hash.get(n.content_hash)
        if candidates and (n.symbol is None) == (candidates[0].symbol is None):
            o = candidates.pop(0)
            matches.append(Match(o, n, _moved_or_renamed(o, n)))
        else:
            still_new.append(n)
    old_left = [o for group in by_hash.values() for o in group]

    # Only symbols pair by similarity; whole-file chunks are matched by path above.
    olds = [(o, s) for o in old_left if o.symbol and len(s := _shingles(o)) >= MIN_SHINGLES]
    news = [(n, s) for n in still_new if n.symbol and len(s := _shingles(n)) >= MIN_SHINGLES]
    pairs = []
    for i, (o, so) in enumerate(olds):
        for j, (n, sn) in enumerate(news):
            if o.language != n.language:
                continue
            score = _jaccard(so, sn)
            if score >= threshold:
                bonus = 0.05 * (o.path == n.path) + 0.05 * (o.symbol == n.symbol)
                pairs.append((score + bonus, score, i, j))
    used_old: set[int] = set()
    used_new: set[int] = set()
    for _, score, i, j in sorted(pairs, reverse=True):
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        o, n = olds[i][0], news[j][0]
        matches.append(Match(o, n, _moved_or_renamed(o, n), round(score, 4)))

    paired_new = {id(news[j][0]) for j in used_new}
    paired_old = {id(olds[i][0]) for i in used_old}
    matches += [Match(None, n, ChangeType.ADDED, 0.0) for n in still_new if id(n) not in paired_new]
    matches += [Match(o, None, ChangeType.DELETED, 0.0) for o in old_left if id(o) not in paired_old]
    return matches


def symbol_ref(chunk: Any) -> str:
    return f"{chunk.path}::{chunk.symbol}" if chunk.symbol else chunk.path


class IndexVersionStore:
    """`VersionStore` over the lineage recorded by a `VersionedIndex`.

    Versions and lineage are loaded when the store is created, so it stays usable (e.g. inside a
    cached pipeline) after the index is closed; only `diff_symbol` reads the index again.
    """

    def __init__(self, index: VersionedIndex) -> None:
        self.index = index
        commits = index.ordered_versions()
        for c in commits:
            index.ensure_lineage(c)
        info, tags, bases = index.commit_info(commits), index.tags(), index.lineage_bases()
        self._versions = [Version(c, c, tags.get(c), bases.get(c), info.get(c, (0, ""))[0]) for c in commits]
        rank = {v.id: i for i, v in enumerate(self._versions)}
        self._nodes: dict[str, list[LineageNode]] = defaultdict(list)
        for r in index.lineage_rows():
            self._nodes[r["lineage_id"]].append(LineageNode(
                r["lineage_id"],
                f"{r['path']}::{r['symbol']}" if r["symbol"] else r["path"],
                r["commit_id"],
                r["occurrence_id"],
                ChangeType(r["change_type"]),
                info.get(r["commit_id"], (0, ""))[1],
            ))
        for nodes in self._nodes.values():
            nodes.sort(key=lambda n: rank.get(n.version_id, -1))

    def versions(self) -> list[Version]:
        return list(self._versions)

    def lineage(self, lineage_id: str) -> list[LineageNode]:
        return list(self._nodes.get(lineage_id, []))

    def diff_symbol(self, symbol: str, a: str, b: str) -> SymbolDiff:
        """Diff one symbol (`path::symbol` or a name) between two indexed versions, following renames."""
        ca, cb = self.index.resolve(a), self.index.resolve(b)
        old, new = self.index.find_lineage_pair(symbol, ca, cb)
        if old is None and new is None:
            change = ChangeType.UNCHANGED
        elif old is None:
            change = ChangeType.ADDED
        elif new is None:
            change = ChangeType.DELETED
        elif old.content_hash == new.content_hash and symbol_ref(old) == symbol_ref(new):
            change = ChangeType.UNCHANGED
        elif old.path != new.path and old.symbol == new.symbol:
            change = ChangeType.MOVED
        elif old.symbol != new.symbol:
            change = ChangeType.RENAMED
        else:
            change = ChangeType.MODIFIED
        ta, tb = (old.text if old else None), (new.text if new else None)
        diff = "".join(difflib.unified_diff(
            (ta or "").splitlines(True), (tb or "").splitlines(True),
            f"{a}:{symbol_ref(old)}" if old else a, f"{b}:{symbol_ref(new)}" if new else b,
        ))
        return SymbolDiff(symbol, ca, cb, change, ta, tb, diff)
