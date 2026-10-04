"""QueryAnalyzer implementations and the factory used by the pipeline."""

from __future__ import annotations

from seraph.config import SeraphConfig
from seraph.query.classifier import LearnedClassifier, RuleClassifier
from seraph.query.decomposition import decompose
from seraph.query.parser import parse
from seraph.types import AnalyzedQuery, SubQuery


class RuleQueryAnalyzer:
    def __init__(self, classifier=None, use_decomposition: bool = False, weighter=None) -> None:
        self.classifier = classifier or RuleClassifier()
        self.use_decomposition = use_decomposition
        self.weighter = weighter

    def analyze(self, query: str) -> AnalyzedQuery:
        p = parse(query)
        qtype, probs = self.classifier.predict(p)
        subs = decompose(p) if self.use_decomposition else [SubQuery(query, "full", 1.0)]
        aq = AnalyzedQuery(
            raw=query,
            text=query,
            query_type=qtype,
            identifiers=p.identifiers,
            entities=p.entities,
            actions=p.actions,
            conditions=p.conditions,
            version_refs=p.version_refs,
            dependency_cues=p.dependency_cues,
            sub_queries=subs,
            features=dict(p.features),
            type_probs=probs,
        )
        if self.weighter is not None:
            aq.weights = self.weighter.weights(aq)
        return aq


def build_analyzer(cfg: SeraphConfig, weighter=None):
    r = cfg.retrieval
    clf = None
    model_path = cfg.cache_path / "query_classifier.pkl"
    if model_path.exists():
        clf = LearnedClassifier.load(model_path)
    rules = RuleQueryAnalyzer(clf, r.use_decomposition, weighter)
    if r.analyzer == "llm":
        from seraph.query.llm import LLMClient, LLMQueryAnalyzer

        return _WithWeights(LLMQueryAnalyzer(LLMClient(cfg.llm, cfg.cache_path), rules), weighter)
    return rules


class _WithWeights:
    """Re-derive weights after the LLM may have changed the query type."""

    def __init__(self, inner, weighter) -> None:
        self.inner, self.weighter = inner, weighter

    def analyze(self, query: str) -> AnalyzedQuery:
        aq = self.inner.analyze(query)
        if self.weighter is not None:
            aq.weights = self.weighter.weights(aq)
        return aq
