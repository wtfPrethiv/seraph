"""Snippet-level structural features (gamma view).

Documents get a sparse binary vector of AST-derived traits (I/O style, control flow, algorithmic
APIs, imports). A multi-label model predicts, from the query text alone, which traits the answer
should have; a chunk's structural score is the Bernoulli log-likelihood ratio of its traits under
the predicted distribution versus the training prior.
"""

from __future__ import annotations

import pickle
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from seraph.types import AnalyzedQuery, Chunk, ScoredChunk, View

_MOD = re.compile(r"\b(?:1000000007|998244353|10\s*\*\*\s*9\s*\+\s*7|1e9\s*\+\s*7)\b")
_MULTITEST = re.compile(
    r"for\s+\w+\s+in\s+range\(\s*int\(\s*(?:input|sys\.stdin\.readline)\(\)|"
    r"^\s*(\w+)\s*=\s*int\(\s*(?:input|sys\.stdin\.readline)\(\)\s*\)\s*$(?=[\s\S]*(?:range\(\s*\1\s*\)|while\s+\1))",
    re.M,
)
_READ_GRID = re.compile(r"\[\s*(?:list\(|input\(|\w+\(input)[^\]]*for\s+\w+\s+in\s+range")

CALL_TRAITS = {
    "sort": {"sorted", "sort"},
    "heapq": {"heappush", "heappop", "heapify", "nlargest", "nsmallest", "heappushpop"},
    "deque": {"deque", "popleft", "appendleft"},
    "bisect": {"bisect", "bisect_left", "bisect_right", "insort"},
    "gcd": {"gcd", "lcm"},
    "counter": {"Counter", "most_common"},
    "defaultdict": {"defaultdict"},
    "memo": {"lru_cache", "cache"},
    "combinatorics": {"permutations", "combinations", "product", "accumulate", "combinations_with_replacement"},
    "string_ops": {"split", "join", "strip", "replace", "find", "count", "startswith", "endswith"},
    "char_codes": {"ord", "chr"},
    "set_ops": {"set", "add", "discard", "union", "intersection"},
    "minmax": {"min", "max"},
    "sum": {"sum"},
    "abs": {"abs"},
    "enumerate_zip": {"enumerate", "zip"},
    "map_int": {"map"},
    "sqrt": {"sqrt", "isqrt", "ceil", "floor", "log", "log2"},
    "pow_mod": {"pow"},
    "setrecursionlimit": {"setrecursionlimit"},
    "regex": {"match", "findall", "sub", "search", "compile", "fullmatch"},
    "stdout_write": {"write"},
    "reversed": {"reversed"},
    "dict_methods": {"items", "keys", "values", "get", "setdefault"},
}
FIXED_TRAITS = (
    "io_input",
    "io_stdin",
    "io_multitest",
    "io_grid",
    "print",
    "loop",
    "loop_nested2",
    "loop_nested3",
    "while_loop",
    "comprehension",
    "recursion",
    "multi_def",
    "class_def",
    "lambda",
    "try_except",
    "global",
    "subscript_2d",
    "slice_reverse",
    "bit_ops",
    "floor_div",
    "float_div",
    "mod_const",
    "modulo",
    "dict_literal",
    "string_format",
    "early_exit",
    "yield",
    *CALL_TRAITS,
)
_CALL_INDEX = {name: trait for trait, names in CALL_TRAITS.items() for name in names}
_LOOPS = {"for_statement", "while_statement"}
_COMPREHENSIONS = {"list_comprehension", "set_comprehension", "dictionary_comprehension", "generator_expression"}


@lru_cache(maxsize=1)
def _parser():
    import tree_sitter
    import tree_sitter_python

    return tree_sitter.Parser(tree_sitter.Language(tree_sitter_python.language()))


def _callee(node, src: bytes) -> tuple[str, str]:
    fn = node.child_by_field_name("function")
    if fn is None:
        return "", ""
    text = src[fn.start_byte : fn.end_byte].decode("utf-8", "replace")
    return text, text.rsplit(".", 1)[-1]


def extract_traits(code: str) -> set[str]:
    """Binary structural traits plus `imp:<module>` entries for a Python snippet."""
    src = code.encode("utf-8", "replace")
    tree = _parser().parse(src)
    traits: set[str] = set()
    defs: list[str] = []
    stack = [(tree.root_node, 0, None)]
    while stack:
        node, depth, func = stack.pop()
        t = node.type
        if t in _LOOPS:
            depth += 1
            traits.add("loop")
            if t == "while_statement":
                traits.add("while_loop")
            if depth >= 2:
                traits.add("loop_nested2")
            if depth >= 3:
                traits.add("loop_nested3")
        elif t in _COMPREHENSIONS:
            traits.add("comprehension")
        elif t == "function_definition":
            name = node.child_by_field_name("name")
            func = src[name.start_byte : name.end_byte].decode() if name else None
            if func:
                defs.append(func)
        elif t == "class_definition":
            traits.add("class_def")
        elif t == "lambda":
            traits.add("lambda")
        elif t == "try_statement":
            traits.add("try_except")
        elif t == "global_statement":
            traits.add("global")
        elif t in ("yield", "yield_expression"):
            traits.add("yield")
        elif t == "dictionary":
            traits.add("dict_literal")
        elif t in ("interpolation", "format_expression"):
            traits.add("string_format")
        elif t == "subscript":
            value = node.child_by_field_name("value")
            if value is not None and value.type == "subscript":
                traits.add("subscript_2d")
        elif t == "slice":
            text = src[node.start_byte : node.end_byte]
            if text.replace(b" ", b"") == b"::-1":
                traits.add("slice_reverse")
        elif t in ("binary_operator", "augmented_assignment"):
            op = node.child_by_field_name("operator")
            op_text = src[op.start_byte : op.end_byte].decode() if op else ""
            op_text = op_text.rstrip("=")
            if op_text in ("&", "|", "^", "<<", ">>"):
                traits.add("bit_ops")
            elif op_text == "//":
                traits.add("floor_div")
            elif op_text == "/":
                traits.add("float_div")
            elif op_text == "%":
                traits.add("modulo")
        elif t in ("import_statement", "import_from_statement"):
            mod = node.child_by_field_name("module_name")
            names = [mod] if mod is not None else node.children_by_field_name("name")
            for m in names:
                top = src[m.start_byte : m.end_byte].decode().split(".")[0].split(" ")[0]
                traits.add(f"imp:{top}")
        elif t == "call":
            full, short = _callee(node, src)
            if short == "input" or full in ("sys.stdin.readline", "stdin.readline"):
                traits.add("io_input" if short == "input" else "io_stdin")
            elif short == "print":
                traits.add("print")
            elif short in ("exit", "quit") or full == "sys.exit":
                traits.add("early_exit")
            elif short == "format":
                traits.add("string_format")
            if short in _CALL_INDEX and not (short == "write" and "stdout" not in full):
                traits.add(_CALL_INDEX[short])
            if func and short == func and full == short:
                traits.add("recursion")
        elif t == "attribute" and src[node.start_byte : node.end_byte].startswith(b"sys.stdin"):
            traits.add("io_stdin")
        for child in node.children:
            stack.append((child, depth, func))
    if len(defs) >= 2:
        traits.add("multi_def")
    if _MOD.search(code):
        traits.add("mod_const")
    if _MULTITEST.search(code):
        traits.add("io_multitest")
    if _READ_GRID.search(code):
        traits.add("io_grid")
    return traits


class StructuralModel:
    """Trait vocabulary + query -> trait probability model."""

    def __init__(self, min_import_df: float = 0.005, C: float = 1.0) -> None:
        self.min_import_df = min_import_df
        self.C = C
        self.vocab: list[str] = []
        self.prior: np.ndarray = np.zeros(0)
        self.vectorizer = None
        self.models: list = []

    def doc_matrix(self, texts: Sequence[str]) -> np.ndarray:
        index = {f: i for i, f in enumerate(self.vocab)}
        x = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in extract_traits(text):
                if f in index:
                    x[row, index[f]] = 1.0
        return x

    def fit(self, queries: Sequence[str], gold_docs: Sequence[str]) -> StructuralModel:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        doc_traits = [extract_traits(d) for d in gold_docs]
        imports = Counter(f for ts in doc_traits for f in ts if f.startswith("imp:"))
        min_count = max(2, int(self.min_import_df * len(gold_docs)))
        self.vocab = list(FIXED_TRAITS) + sorted(f for f, c in imports.items() if c >= min_count)
        y = self.doc_matrix_from_traits(doc_traits)
        self.prior = np.clip(y.mean(axis=0), 1e-3, 1 - 1e-3)
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=50000)
        xq = self.vectorizer.fit_transform(queries)
        self.models = []
        for j in range(len(self.vocab)):
            col = y[:, j]
            if col.min() == col.max():
                self.models.append(None)
                continue
            self.models.append(LogisticRegression(C=self.C, max_iter=1000).fit(xq, col))
        return self

    def doc_matrix_from_traits(self, doc_traits: Iterable[set[str]]) -> np.ndarray:
        index = {f: i for i, f in enumerate(self.vocab)}
        rows = list(doc_traits)
        y = np.zeros((len(rows), len(self.vocab)), dtype=np.float32)
        for r, ts in enumerate(rows):
            for f in ts:
                if f in index:
                    y[r, index[f]] = 1.0
        return y

    def predict_proba(self, queries: Sequence[str]) -> np.ndarray:
        assert self.vectorizer is not None, "model is not fitted"
        xq = self.vectorizer.transform(queries)
        p = np.tile(self.prior, (len(queries), 1))
        for j, m in enumerate(self.models):
            if m is not None:
                p[:, j] = m.predict_proba(xq)[:, 1]
        return np.clip(p, 1e-3, 1 - 1e-3)

    def llr_weights(self, queries: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Per-query (w, b) so that score(doc) = x_doc @ w + b is the trait log-likelihood ratio."""
        p = self.predict_proba(queries)
        on = np.log(p / self.prior)
        off = np.log((1 - p) / (1 - self.prior))
        return on - off, off.sum(axis=1)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> StructuralModel:
        with open(path, "rb") as f:
            return pickle.load(f)


class StructuralScorer:
    """Gamma view: rescores a candidate pool by predicted-trait agreement."""

    name = View.STRUCTURAL.value

    def __init__(self, model: StructuralModel) -> None:
        self.model = model
        self._rows: dict[str, int] = {}
        self._x = np.zeros((0, len(model.vocab)), dtype=np.float32)

    def index(self, chunks: Iterable[Chunk]) -> None:
        chunks = list(chunks)
        self._rows = {c.chunk_hash: i for i, c in enumerate(chunks)}
        self._x = self.model.doc_matrix([c.text for c in chunks])

    def rescore(
        self, queries: Sequence[AnalyzedQuery], candidates: Sequence[Sequence[ScoredChunk]]
    ) -> list[list[ScoredChunk]]:
        w, _ = self.model.llr_weights([q.raw or q.text for q in queries])
        out = []
        for i, cands in enumerate(candidates):
            rows = [self._rows.get(h.chunk_hash) for h in cands]
            known = [(h, r) for h, r in zip(cands, rows, strict=True) if r is not None]
            if not known:
                out.append([])
                continue
            s = self._x[[r for _, r in known]] @ w[i]
            order = np.argsort(-s, kind="stable")
            out.append([ScoredChunk(known[j][0].chunk, float(s[j]), {self.name: float(s[j])}) for j in order])
        return out


DEFAULT_MODEL_PATH = "experiments/models/structural.pkl"


def train_on_apps(cache_dir: str = ".seraph_cache", out: str = DEFAULT_MODEL_PATH, C: float = 1.0) -> StructuralModel:
    """Fit on the AppsRetrieval `fit` split (train minus dev) only."""
    from seraph.evaluation.coir import load_apps

    data = load_apps(cache_dir)
    s = data.split("fit")
    queries, docs = [], []
    for qid, rels in s.qrels.items():
        for doc_id, rel in rels.items():
            if rel > 0 and doc_id in data.corpus:
                queries.append(s.queries[qid])
                docs.append(data.corpus[doc_id])
    model = StructuralModel(C=C).fit(queries, docs)
    model.save(out)
    return model
