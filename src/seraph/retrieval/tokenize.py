"""Code-aware tokenizer shared by BM25 and query analysis."""

from __future__ import annotations

import re
from functools import lru_cache

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "she", "so", "such", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "were", "which", "while", "who", "will", "with", "you", "your", "our", "can", "do", "does", "not", "no"]
)


def split_identifier(ident: str) -> list[str]:
    """`parseHTTPResponse_v2` -> [parse, http, response, v, 2]."""
    parts: list[str] = []
    for piece in ident.split("_"):
        parts.extend(p.lower() for p in _CAMEL.findall(piece))
    return [p for p in parts if p]


@lru_cache(maxsize=1)
def _stemmer():
    import Stemmer

    return Stemmer.Stemmer("english")


def tokenize(text: str, stem: bool = True, keep_identifiers: bool = True) -> list[str]:
    out: list[str] = []
    for tok in _IDENT.findall(text):
        sub = split_identifier(tok) if not tok.isdigit() else [tok]
        lowered = tok.lower()
        if keep_identifiers and len(sub) > 1:
            out.append(lowered)
        out.extend(s for s in sub if s not in STOPWORDS)
    if stem:
        out = _stemmer().stemWords(out)
    return out
