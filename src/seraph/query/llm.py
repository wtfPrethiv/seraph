"""Optional LLM query analyzer (any OpenAI-compatible endpoint) with an on-disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from seraph.config import LLMConfig
from seraph.types import AnalyzedQuery, QueryType, SubQuery

log = logging.getLogger(__name__)

PROMPT = """You analyze search queries for a code retrieval engine.
Return ONLY a JSON object with keys:
  query_type: one of LEXICAL, SEMANTIC, STRUCTURAL, DEPENDENCY, EVOLUTIONARY, MIXED
  identifiers: code identifiers mentioned
  entities: domain nouns (data structures, objects)
  actions: verbs describing what the code does
  conditions: constraints or conditions
  version_refs: versions, tags, commits, or time references
  sub_queries: list of {"text": str, "facet": str} short self-contained retrieval queries
Query:
"""


class LLMClient:
    def __init__(self, cfg: LLMConfig, cache_dir: str | Path) -> None:
        self.cfg = cfg
        self.cache = Path(cache_dir) / "llm"
        self.cache.mkdir(parents=True, exist_ok=True)

    def complete(self, prompt: str) -> str:
        key = hashlib.sha1(f"{self.cfg.model}|{prompt}".encode()).hexdigest()
        path = self.cache / f"{key}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        from openai import OpenAI

        client = OpenAI(base_url=self.cfg.base_url, api_key=os.environ.get(self.cfg.api_key_env, "none"))
        resp = client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        path.write_text(text, encoding="utf-8")
        return text


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1]) if start >= 0 and end > start else {}


class LLMQueryAnalyzer:
    """Falls back to the rule analyzer on any failure, so it never blocks search."""

    def __init__(self, client: LLMClient, fallback) -> None:
        self.client = client
        self.fallback = fallback

    def analyze(self, query: str) -> AnalyzedQuery:
        base = self.fallback.analyze(query)
        try:
            data = _extract_json(self.client.complete(PROMPT + query[:4000]))
        except Exception as e:  # network, auth, bad JSON
            log.warning("LLM analyzer failed, using rules: %s", e)
            return base
        qt = data.get("query_type", base.query_type)
        base.query_type = QueryType(qt) if qt in QueryType.__members__ else base.query_type
        for key in ("identifiers", "entities", "actions", "conditions", "version_refs"):
            if isinstance(data.get(key), list):
                setattr(base, key, [str(x) for x in data[key]])
        subs = [
            SubQuery(s["text"], s.get("facet", "clause"), 0.7)
            for s in data.get("sub_queries", [])
            if isinstance(s, dict) and s.get("text")
        ]
        if subs:
            base.sub_queries = [SubQuery(query, "full", 1.0), *subs]
        return base
