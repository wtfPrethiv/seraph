"""Query-type classification: rules by default, optional logistic regression."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from seraph.query.parser import ParsedQuery, parse
from seraph.types import QueryType

TYPES = [t for t in QueryType if t != QueryType.MIXED]


def rule_scores(p: ParsedQuery) -> dict[QueryType, float]:
    f = p.features
    short = f["log_len"] < 2.6  # ~12 words
    s = {
        QueryType.EVOLUTIONARY: 1.5 * f["n_version_refs"] + 1.2 * f["n_evolution_cues"],
        QueryType.DEPENDENCY: 2.5 * f["n_dependency_cues"],
        QueryType.LEXICAL: 3.0 * f["identifier_ratio"] + (0.8 if short and f["n_identifiers"] else 0.0),
        QueryType.STRUCTURAL: 0.7 * f["n_structural_cues"] + 0.5 * f["has_io_sections"],
        QueryType.SEMANTIC: 1.0 + 0.2 * f["n_actions"],
    }
    return s


class RuleClassifier:
    mixed_margin = 0.25

    def predict(self, p: ParsedQuery) -> tuple[QueryType, dict[str, float]]:
        s = rule_scores(p)
        z = np.array([s[t] for t in TYPES])
        probs = np.exp(z - z.max())
        probs /= probs.sum()
        pd = {t.value: float(v) for t, v in zip(TYPES, probs, strict=True)}
        order = np.argsort(-probs)
        top, second = probs[order[0]], probs[order[1]]
        non_semantic_strong = sum(
            1 for t in TYPES if t != QueryType.SEMANTIC and s[t] >= 1.0
        )
        if non_semantic_strong >= 2 and top - second < self.mixed_margin:
            return QueryType.MIXED, pd
        return TYPES[order[0]], pd


class LearnedClassifier:
    """TF-IDF + parser features -> logistic regression."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=2000, C=4.0)
        self.feature_keys: list[str] = []

    def _x(self, parsed: Sequence[ParsedQuery], fit: bool = False):
        from scipy.sparse import csr_matrix, hstack

        texts = [p.text for p in parsed]
        tf = self.vec.fit_transform(texts) if fit else self.vec.transform(texts)
        if fit:
            self.feature_keys = sorted(parsed[0].features)
        dense = np.array([[p.features[k] for k in self.feature_keys] for p in parsed])
        return hstack([tf, csr_matrix(dense)]).tocsr()

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> LearnedClassifier:
        parsed = [parse(t) for t in texts]
        self.clf.fit(self._x(parsed, fit=True), list(labels))
        return self

    def predict(self, p: ParsedQuery) -> tuple[QueryType, dict[str, float]]:
        probs = self.clf.predict_proba(self._x([p]))[0]
        pd = {str(c): float(v) for c, v in zip(self.clf.classes_, probs, strict=True)}
        return QueryType(self.clf.classes_[int(np.argmax(probs))]), pd

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> LearnedClassifier:
        return pickle.loads(Path(path).read_bytes())
