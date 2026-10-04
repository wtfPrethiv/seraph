import pytest

from seraph.query.analyzer import RuleQueryAnalyzer
from seraph.query.classifier import LearnedClassifier, RuleClassifier
from seraph.query.decomposition import decompose
from seraph.query.llm import LLMQueryAnalyzer
from seraph.query.parser import parse, split_sections
from seraph.types import QueryType

CP_PROBLEM = """You are given an array of n integers. Find the maximum sum of a subarray.

-----Input-----
The first line contains n ($1 \\le n \\le 10^5$). The second line contains the array.

-----Output-----
Print the maximum sum.

-----Examples-----
Input
3
1 -2 3
Output
3
"""


def test_parser_extracts_parts():
    p = parse("Where does `parse_config` call readFile() in cfg.loader before v1.2?")
    assert {"parse_config", "readFile", "cfg.loader"} <= set(p.identifiers)
    assert "v1.2" in p.version_refs
    assert "before" in p.evolution_cues
    p2 = parse(CP_PROBLEM)
    assert "find" in p2.actions and "array" in p2.entities
    assert any("10^5" in c for c in p2.conditions)
    assert p2.features["has_io_sections"] == 1.0


def test_split_sections():
    s = split_sections(CP_PROBLEM)
    assert set(s) >= {"statement", "input", "output", "example"}
    assert s["statement"].startswith("You are given")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("who calls parse_config", QueryType.DEPENDENCY),
        ("how did the retry logic change since v2.0", QueryType.EVOLUTIONARY),
        ("HttpClient.send_request", QueryType.LEXICAL),
        ("function that validates email addresses", QueryType.SEMANTIC),
        ("recursive dfs with memoization over a grid", QueryType.STRUCTURAL),
    ],
)
def test_rule_classifier(query, expected):
    qt, probs = RuleClassifier().predict(parse(query))
    assert qt == expected
    assert abs(sum(probs.values()) - 1) < 1e-6


def test_mixed_query():
    qt, _ = RuleClassifier().predict(parse("which callers of load_config changed since v1.0 and imports json"))
    assert qt in (QueryType.MIXED, QueryType.EVOLUTIONARY, QueryType.DEPENDENCY)


def test_decompose_cp_problem():
    subs = decompose(parse(CP_PROBLEM))
    facets = [s.facet for s in subs]
    assert facets[0] == "full" and "core" in facets and "input" in facets and "output" in facets
    assert all("Examples" not in s.text for s in subs if s.facet == "core")


def test_decompose_clauses():
    subs = decompose(parse("open the database connection; then retry the failed request"))
    assert sum(s.facet == "clause" for s in subs) == 2


def test_learned_classifier_roundtrip(tmp_path):
    texts = ["who calls foo", "callers of bar", "what changed since v1", "history of baz before v2",
             "sort a list of numbers", "compute the average of values"]
    labels = ["DEPENDENCY", "DEPENDENCY", "EVOLUTIONARY", "EVOLUTIONARY", "SEMANTIC", "SEMANTIC"]
    clf = LearnedClassifier().fit(texts, labels)
    clf.save(tmp_path / "c.pkl")
    qt, _ = LearnedClassifier.load(tmp_path / "c.pkl").predict(parse("who calls qux"))
    assert qt == QueryType.DEPENDENCY


class _BrokenClient:
    def complete(self, prompt):
        raise ConnectionError("offline")


class _JsonClient:
    def complete(self, prompt):
        return 'sure: {"query_type": "DEPENDENCY", "identifiers": ["foo"], "sub_queries": [{"text": "foo callers"}]}'


def test_llm_analyzer_fallback_and_parse():
    rules = RuleQueryAnalyzer()
    assert LLMQueryAnalyzer(_BrokenClient(), rules).analyze("sort numbers").query_type == QueryType.SEMANTIC
    aq = LLMQueryAnalyzer(_JsonClient(), rules).analyze("what uses foo")
    assert aq.query_type == QueryType.DEPENDENCY and aq.identifiers == ["foo"]
    assert [s.text for s in aq.sub_queries] == ["what uses foo", "foo callers"]
