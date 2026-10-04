"""Split a query into sub-objectives, each retrieved separately and recombined."""

from __future__ import annotations

import re

from seraph.query.parser import ParsedQuery
from seraph.types import SubQuery

FACET_WEIGHTS = {"full": 1.0, "core": 1.0, "input": 0.4, "output": 0.4, "constraints": 0.2, "identifier": 0.6, "clause": 0.7}
_CLAUSE_SPLIT = re.compile(r"\s*(?:;|\band then\b|\band also\b|\bthen\b)\s*", re.I)


def decompose(p: ParsedQuery, max_chars: int = 2000) -> list[SubQuery]:
    """Always keeps the full query; adds facet sub-queries when the query has structure."""
    subs = [SubQuery(p.text, "full", FACET_WEIGHTS["full"])]
    sec = p.sections
    if "input" in sec or "output" in sec:
        core = sec.get("statement", "")
        if core and core != p.text:
            subs.append(SubQuery(core[:max_chars], "core", FACET_WEIGHTS["core"]))
        for facet in ("input", "output"):
            if sec.get(facet):
                subs.append(SubQuery(sec[facet][:max_chars], facet, FACET_WEIGHTS[facet]))
        cons = [c for c in p.conditions if len(c) < 300]
        if cons:
            subs.append(SubQuery(" ".join(cons)[:max_chars], "constraints", FACET_WEIGHTS["constraints"]))
        return subs

    clauses = [c for c in _CLAUSE_SPLIT.split(p.text) if len(c.split()) >= 3]
    if len(clauses) > 1:
        subs += [SubQuery(c, "clause", FACET_WEIGHTS["clause"]) for c in clauses]
    if p.identifiers and len(p.text.split()) > len(p.identifiers) * 3:
        subs.append(SubQuery(" ".join(p.identifiers), "identifier", FACET_WEIGHTS["identifier"]))
    return subs
