import subprocess
from pathlib import Path

import pytest

from seraph.config import SeraphConfig
from seraph.index import VersionedIndex
from seraph.service import IndexChunkStore, search_index


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Seraph Test")
    (tmp_path / "config.py").write_text(
        "def parse_config(path):\n    return json.load(open(path))\n\n"
        "def normalize_input(text):\n    return text.strip().lower()\n"
    )
    (tmp_path / "net.py").write_text("def open_socket(host, port):\n    return socket.create_connection((host, port))\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    old = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "config.py").write_text(
        "def parse_config(path):\n    data = yaml.safe_load(open(path))\n    validate_schema(data)\n    return data\n\n"
        "def normalize_input(text):\n    return text.strip().lower()\n"
    )
    git(tmp_path, "commit", "-qam", "switch config to yaml with schema validation")
    return tmp_path, old, git(tmp_path, "rev-parse", "HEAD")


def test_store_maps_index_chunks(repo):
    path, old, new = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        index.index_commit(old)
        index.index_commit(new)
        store = IndexChunkStore.from_index(index, "HEAD", include_history=True)
        head = list(store.iter_chunks("HEAD"))
        assert {c.symbol for c in head} == {"parse_config", "normalize_input", "open_socket"}
        assert all(c.version_id == new for c in head)
        assert len(list(store.iter_chunks(old[:10]))) == 3 and len(list(store.iter_chunks("*"))) == 6
        parse = next(c for c in head if c.symbol == "parse_config")
        old_parse = next(c for c in store.iter_chunks(old[:10]) if c.symbol == "parse_config")
        assert parse.lineage_id == old_parse.lineage_id is not None and store.get(parse.chunk_hash) == parse
        changes = store.changed_since(old)
        assert [store.get(o).symbol for o in changes.added] == ["parse_config"]
        assert len(changes.unchanged) == 2


def test_search_index_uses_pipeline_and_versions(repo):
    path, old, new = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        out = search_index(index, "parseConfig yaml schema", "HEAD", 3, cfg=SeraphConfig())
        assert out["resolved_commit"] == new
        top = out["results"][0]
        assert top["symbol"] == "parse_config" and top["commit"] == new and "lexical" in top["retrieval_scores"]
        assert {"occurrence_id", "path", "start_line", "end_line", "text", "content_hash"} <= set(top)
        old_out = search_index(index, "json load config", old, 3, cfg=SeraphConfig())
        assert old_out["results"][0]["commit"] == old and "json.load" in old_out["results"][0]["text"]
        hist = search_index(index, "json load config", "HEAD", 5, include_history=True, cfg=SeraphConfig())
        assert {r["commit"] for r in hist["results"]} == {old, new}
        assert search_index(index, "   ", "HEAD", 3, cfg=SeraphConfig())["results"] == []


def test_history_search_collapses_identical_code_across_versions(repo):
    path, old, new = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        index.index_commit(old)
        index.index_commit(new)
        out = search_index(index, "normalize input text", "HEAD", 5, include_history=True, cfg=SeraphConfig())
        changed = search_index(index, "parse config yaml json", "HEAD", 5, include_history=True, cfg=SeraphConfig())
    normalize = [r for r in out["results"] if r["symbol"] == "normalize_input"]
    assert len(normalize) == 1
    assert sorted(normalize[0]["versions"]) == sorted([old, new])
    parse = [r for r in changed["results"] if r["symbol"] == "parse_config"]
    assert {r["commit"] for r in parse} == {old, new}
