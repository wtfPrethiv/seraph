import subprocess
from pathlib import Path

import pytest

from seraph.config import RetrievalConfig, SeraphConfig
from seraph.graph.builder import Ref, build_graph, extract_refs
from seraph.graph.traversal import find_dependencies
from seraph.index import VersionedIndex, _chunks
from seraph.service import repo_graph, search_index
from seraph.types import Direction, EdgeKind


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit_files(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


PY_FILES = {
    "pkg/__init__.py": "",
    "pkg/io.py": "def read_file(path):\n    return open(path).read()\n",
    "pkg/cfg.py": (
        "import json\n\nfrom .io import read_file\n\n\n"
        "def parse_config(path):\n    return json.loads(read_file(path))\n"
    ),
    "app.py": (
        "from pkg import cfg\nfrom pkg.cfg import parse_config\n\n\n"
        "class Base:\n    pass\n\n\n"
        "class App(Base):\n    def load(self):\n        self.config = cfg.parse_config(self.path)\n"
        "        return self.helper()\n\n    def helper(self):\n        return parse_config\n\n\n"
        "def main():\n    return App().load()\n"
    ),
}


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Seraph Test")
    first = commit_files(tmp_path, PY_FILES, "initial")
    return tmp_path, first


def edges_of(graph, kind):
    return {(s, d) for s, d, k in graph.edges() if k == kind}


def test_python_calls_imports_inheritance_and_references(repo):
    path, first = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        graph = repo_graph(index, first)
    calls = edges_of(graph, EdgeKind.CALLS)
    assert ("pkg/cfg.py::parse_config", "pkg/io.py::read_file") in calls  # relative from-import
    assert ("app.py::App", "pkg/cfg.py::parse_config") in calls  # module imported from a package
    assert ("app.py::main", "app.py::App") in calls
    assert not any(d.startswith("json") for _, d in calls)  # stdlib stays out of the graph
    assert edges_of(graph, EdgeKind.INHERITS) == {("app.py::App", "app.py::Base")}
    assert ("app.py::App", "pkg/cfg.py::parse_config") in edges_of(graph, EdgeKind.REFERENCES)
    imports = edges_of(graph, EdgeKind.IMPORTS)
    assert {("app.py", "pkg/cfg.py"), ("app.py", "pkg/cfg.py::parse_config"), ("pkg/cfg.py", "pkg/io.py::read_file")} <= imports
    assert graph.resolve("parse_config") == ["pkg/cfg.py::parse_config"]
    assert graph.chunk_for_symbol("app.py::App") is not None

    chains = find_dependencies(graph, "app.py::main", Direction.OUT, {EdgeKind.CALLS})
    assert ["app.py::main", "app.py::App", "pkg/cfg.py::parse_config", "pkg/io.py::read_file"] in [c.symbols for c in chains]
    callers = find_dependencies(graph, "pkg/io.py::read_file", Direction.IN, {EdgeKind.CALLS}, max_depth=1)
    assert [c.symbols[-1] for c in callers] == ["pkg/cfg.py::parse_config"]


def test_refs_of_unchanged_files_are_reused(repo):
    path, first = repo
    second = commit_files(path, {"app.py": "from pkg.cfg import parse_config\n\n\ndef run():\n    return parse_config('x')\n"}, "slim app")
    with VersionedIndex(path, path / "idx.sqlite") as index:
        index.index_commit(first)
        stats = index.index_commit(second)
        assert stats.parsed_files == 1
        graph = repo_graph(index, second)
    calls = edges_of(graph, EdgeKind.CALLS)
    assert ("pkg/cfg.py::parse_config", "pkg/io.py::read_file") in calls
    assert ("app.py::run", "pkg/cfg.py::parse_config") in calls
    assert not any(s.startswith("app.py::App") for s, _ in calls)


def test_refs_are_backfilled_for_commits_indexed_before_the_graph(repo):
    path, first = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        index.index_commit(first)
        index.conn.execute("DELETE FROM refs")
        index.conn.execute("DELETE FROM derived")
        index.conn.commit()
        refs = list(index.iter_refs(first))
    assert ("pkg/cfg.py", Ref("parse_config", "calls", "read_file")) in refs


def test_graph_expansion_reaches_callees_through_the_service(repo):
    path, _ = repo
    cfg = SeraphConfig(retrieval=RetrievalConfig(use_graph_expansion=True))
    with VersionedIndex(path, path / "idx.sqlite") as index:
        out = search_index(index, "main entry point", "HEAD", 5, cfg=cfg)
    by_symbol = {r["symbol"]: r for r in out["results"]}
    assert "graph" in by_symbol["App"]["retrieval_scores"]


def test_unknown_receivers_resolve_only_to_a_unique_method():
    class C:
        def __init__(self, path, symbol):
            self.path, self.symbol, self.occurrence_id = path, symbol, f"{path}:{symbol}"

    chunks = [C("a.ts", "Runner.run"), C("b.ts", "Job.run"), C("b.ts", "Job.stop"), C("c.ts", "go")]
    refs = [("c.ts", Ref("go", "calls", "?.stop")), ("c.ts", Ref("go", "calls", "?.run"))]
    assert edges_of(build_graph(chunks, refs), EdgeKind.CALLS) == {("c.ts::go", "b.ts::Job.stop")}


def test_typescript_graph():
    pytest.importorskip("tree_sitter_typescript")
    util = "export function readFile(p: string) {\n  return p;\n}\n"
    runner = (
        'import { readFile as rf } from "./util";\nimport * as util from "./util";\n\n'
        "export class Base {\n  x = 1;\n}\n\n"
        "export class Runner extends Base {\n  run(task: string) {\n    return this.prep(rf(task));\n  }\n\n"
        "  prep(t: string) {\n    return util.readFile(t);\n  }\n}\n"
    )
    graph = _graph_from_sources({"src/util.ts": util, "src/runner.ts": runner})
    assert edges_of(graph, EdgeKind.CALLS) >= {
        ("src/runner.ts::Runner.run", "src/runner.ts::Runner.prep"),
        ("src/runner.ts::Runner.run", "src/util.ts::readFile"),
        ("src/runner.ts::Runner.prep", "src/util.ts::readFile"),
    }
    assert edges_of(graph, EdgeKind.INHERITS) == {("src/runner.ts::Runner", "src/runner.ts::Base")}
    assert ("src/runner.ts", "src/util.ts::readFile") in edges_of(graph, EdgeKind.IMPORTS)


def test_go_graph():
    pytest.importorskip("tree_sitter_go")
    load = "package util\n\nfunc Load(p string) string {\n\treturn p\n}\n"
    main = (
        'package main\n\nimport "example.com/proj/util"\n\ntype Server struct{}\n\n'
        'func (s *Server) Start() error {\n\tutil.Load("x")\n\treturn nil\n}\n\n'
        "func main() {\n\ts := &Server{}\n\ts.Start()\n\thelper()\n}\n\nfunc helper() {\n}\n"
    )
    graph = _graph_from_sources({"util/load.go": load, "main.go": main})
    assert edges_of(graph, EdgeKind.CALLS) >= {
        ("main.go::Server.Start", "util/load.go::Load"),
        ("main.go::main", "main.go::Server.Start"),
        ("main.go::main", "main.go::helper"),
    }
    assert ("main.go", "util/load.go") in edges_of(graph, EdgeKind.IMPORTS)


def _graph_from_sources(files: dict[str, str]):
    class C:
        def __init__(self, path, symbol, kind):
            self.path, self.symbol, self.kind, self.occurrence_id = path, symbol, kind, f"{path}:{symbol}"

    chunks, refs = [], []
    for path, text in files.items():
        spans = _chunks(path, text)
        chunks += [C(path, s[0], s[1]) for s in spans]
        refs += [(path, r) for r in extract_refs(path, text, spans)]
    return build_graph(chunks, refs)
