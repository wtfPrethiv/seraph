"""Rule-based query parser: identifiers, entities, actions, conditions, version refs, cues."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_BACKTICK = re.compile(r"`([^`]+)`")
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_CAMEL = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]*)+\b")
_PASCAL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
_SNAKE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_CALL = re.compile(r"\b([A-Za-z_]\w*)\(\)")
_WORD = re.compile(r"[A-Za-z]+")

_SEMVER = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
_TAG = re.compile(r"\bv\d+(?:\.\d+)*\b")
_SHA = re.compile(r"\b[0-9a-f]{7,40}\b")
_CONSTRAINT = re.compile(
    r"(?:\\le(?:q)?|\\ge(?:q)?|<=|>=|≤|≥|\bat most\b|\bat least\b|\bno more than\b|\bnot exceed)",
    re.I,
)
_CONDITION_WORDS = re.compile(r"\b(?:if|when|unless|whenever|only if|otherwise|provided that)\b", re.I)
_SECTION = re.compile(r"-{3,}\s*(input|output|examples?|note|constraints)\s*-{3,}", re.I)

ACTIONS = {
    "sort", "find", "count", "compute", "calculate", "parse", "read", "write", "print", "return",
    "check", "validate", "merge", "split", "reverse", "search", "insert", "delete", "remove",
    "update", "call", "handle", "convert", "generate", "minimize", "maximize", "load", "save",
    "send", "fetch", "build", "create", "initialize", "connect", "render", "decode", "encode",
    "serialize", "deserialize", "cache", "retry", "log", "authenticate", "determine", "choose",
    "select", "construct", "restore", "replace", "swap", "move", "visit", "traverse", "match",
    "compare", "rotate", "partition", "assign", "distribute", "simulate", "output", "answer",
}
ENTITIES = {
    "array", "string", "graph", "tree", "matrix", "grid", "permutation", "sequence", "integer",
    "number", "query", "queries", "edge", "edges", "vertex", "vertices", "node", "nodes", "path",
    "interval", "subarray", "substring", "subsequence", "palindrome", "prime", "bit", "bits",
    "binary", "set", "map", "dictionary", "list", "stack", "queue", "heap", "segment", "point",
    "points", "circle", "rectangle", "polygon", "coin", "coins", "cost", "weight", "digit",
    "digits", "character", "characters", "word", "words", "file", "config", "request", "response",
    "user", "token", "session", "database", "table", "client", "server", "handler", "cache",
    "function", "method", "class", "module", "test", "pairs", "pair", "game", "board", "city",
    "cities", "road", "roads", "time", "modulo", "divisor", "divisors", "fraction", "multiset",
}
STRUCTURAL_CUES = {
    "loop", "loops", "recursion", "recursive", "nested", "decorator", "iterator", "generator",
    "inheritance", "pattern", "implements", "implementation", "dfs", "bfs", "dp", "dijkstra",
    "memoization", "memoize", "greedy", "bitmask", "backtracking", "sieve", "two-pointer",
    "sliding", "prefix", "suffix", "union-find", "dsu", "trie", "segment", "fenwick", "heapq",
    "exception", "try", "except", "async", "await", "lambda", "callback", "singleton", "factory",
}
DEPENDENCY_PHRASES = (
    "calls", "called by", "callers", "callees", "caller", "callee", "who uses", "uses of",
    "depends on", "dependency", "dependencies", "imports", "imported by", "invokes", "invoked by",
    "references", "referenced by", "subclass", "subclasses", "inherits", "inherit from",
    "overrides", "call chain", "call graph", "where is", "used by",
)
EVOLUTION_PHRASES = (
    "before", "since", "previous version", "previously", "changed", "change", "history",
    "used to", "regression", "release", "renamed", "refactor", "refactored", "introduced",
    "deprecated", "removed in", "added in", "commit", "diff", "older version", "evolution",
    "how did", "over time", "last version", "between versions",
)


@dataclass
class ParsedQuery:
    text: str
    identifiers: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    version_refs: list[str] = field(default_factory=list)
    dependency_cues: list[str] = field(default_factory=list)
    evolution_cues: list[str] = field(default_factory=list)
    structural_cues: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)


def _stem(word: str) -> str:
    w = word.lower()
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            if base in ACTIONS or base in ENTITIES:
                return base
            if suf == "ed" and base + "e" in ACTIONS:
                return base + "e"
    return w


def _dedup(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in xs if not (x in seen or seen.add(x))]


def _phrases(text_lower: str, phrases: tuple[str, ...]) -> list[str]:
    return [p for p in phrases if re.search(rf"\b{re.escape(p)}\b", text_lower)]


def split_sections(text: str) -> dict[str, str]:
    """Split competitive-programming statements on `-----Input-----` style markers."""
    parts = _SECTION.split(text)
    if len(parts) == 1:
        return {"statement": text.strip()}
    out = {"statement": parts[0].strip()}
    for name, body in zip(parts[1::2], parts[2::2], strict=False):
        key = name.lower().rstrip("s") if name.lower().startswith("example") else name.lower()
        out[key] = (out.get(key, "") + "\n" + body).strip()
    return out


def parse(text: str) -> ParsedQuery:
    lower = text.lower()
    idents = [m.group(1) for m in _BACKTICK.finditer(text)]
    idents += [m.group(1) for m in _CALL.finditer(text)]
    for rx in (_DOTTED, _CAMEL, _PASCAL, _SNAKE):
        idents += [m.group(0) for m in rx.finditer(text)]
    idents = _dedup([i for i in idents if not re.fullmatch(r"[\d.]+", i)])

    words = [_stem(w) for w in _WORD.findall(text)]
    actions = _dedup([w for w in words if w in ACTIONS])
    entities = _dedup([w for w in words if w in ENTITIES])
    struct = _dedup([w for w in (x.lower() for x in _WORD.findall(text)) if w in STRUCTURAL_CUES])

    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    conditions = _dedup(
        [s.strip() for s in sentences if _CONSTRAINT.search(s) or _CONDITION_WORDS.search(s)]
    )[:20]

    version_refs = _dedup([m.group(0) for m in _TAG.finditer(text)] + [m.group(0) for m in _SHA.finditer(lower)])
    if not version_refs:
        version_refs = [m.group(0) for m in _SEMVER.finditer(text) if "v" in m.group(0)]
    dep = _phrases(lower, DEPENDENCY_PHRASES)
    evo = _phrases(lower, EVOLUTION_PHRASES)
    sections = split_sections(text)

    n_words = max(len(words), 1)
    code_like = sum(1 for t in text.split() if re.search(r"[_().\[\]{}=<>]", t))
    p = ParsedQuery(
        text=text,
        identifiers=idents,
        entities=entities,
        actions=actions,
        conditions=conditions,
        version_refs=version_refs,
        dependency_cues=dep,
        evolution_cues=evo,
        structural_cues=struct,
        sections=sections,
    )
    p.features = {
        "log_len": math.log1p(n_words),
        "n_identifiers": float(len(idents)),
        "identifier_ratio": len(idents) / n_words,
        "code_token_ratio": code_like / max(len(text.split()), 1),
        "n_actions": float(len(actions)),
        "n_entities": float(len(entities)),
        "n_conditions": float(len(conditions)),
        "n_version_refs": float(len(version_refs)),
        "n_dependency_cues": float(len(dep)),
        "n_evolution_cues": float(len(evo)),
        "n_structural_cues": float(len(struct)),
        "has_io_sections": float("input" in sections or "output" in sections),
        "math_density": (text.count("$") + len(_CONSTRAINT.findall(text))) / n_words,
        "is_question": float(bool(re.match(r"\s*(how|what|where|which|who|why|when)\b", lower))),
    }
    return p
