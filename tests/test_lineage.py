import subprocess
from pathlib import Path

import pytest

from seraph.config import SeraphConfig
from seraph.index import VersionedIndex
from seraph.lineage import IndexVersionStore, match_symbols
from seraph.service import search_index
from seraph.types import ChangeType


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


LOAD = (
    "def load_config(path):\n    with open(path) as handle:\n        data = json.load(handle)\n"
    "    validate(data)\n    return data\n\n\n"
)
READ = (
    "def read_config(path):\n    with open(path) as handle:\n        data = json.load(handle)\n"
    "    validate(data)\n    data.setdefault(\"debug\", False)\n    return data\n\n\n"
)
SLUG = "def slugify(text):\n    return text.lower().replace(' ', '-')\n"


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Seraph Test")
    (tmp_path / "cfg.py").write_text(LOAD + "def helper(x):\n    return x + 1\n")
    (tmp_path / "util.py").write_text(SLUG)
    (tmp_path / "old.py").write_text("def legacy():\n    return 42\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    v1 = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "cfg.py").write_text(READ + "def helper(x):\n    return x + 2\n\n\ndef brand_new():\n    return 'fresh'\n")
    (tmp_path / "text").mkdir()
    git(tmp_path, "mv", "util.py", "text/util.py")
    git(tmp_path, "rm", "-q", "old.py")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "rename load_config to read_config and add debug default")
    return tmp_path, v1, git(tmp_path, "rev-parse", "HEAD")


def changes(index, commit):
    rows = index.conn.execute("SELECT * FROM lineage WHERE commit_id=?", (commit,)).fetchall()
    return {r["symbol"]: (r["change_type"], r["lineage_id"]) for r in rows}


def assert_linked(index, v1, v2):
    before, after = changes(index, v1), changes(index, v2)
    assert {s: c for s, (c, _) in before.items()} == dict.fromkeys(["load_config", "helper", "slugify", "legacy"], "added")
    assert {s: c for s, (c, _) in after.items()} == {
        "read_config": "renamed", "helper": "modified", "slugify": "moved", "brand_new": "added", "legacy": "deleted",
    }
    assert after["read_config"][1] == before["load_config"][1]
    assert after["slugify"][1] == before["slugify"][1]
    assert after["legacy"][1] == before["legacy"][1]


def test_renames_moves_and_deletions_keep_their_lineage(repo):
    path, v1, v2 = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        index.index_commit(v1)
        index.index_commit(v2)
        assert_linked(index, v1, v2)


def test_indexing_an_older_commit_later_relinks_descendants(repo):
    path, v1, v2 = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        index.index_commit(v2)
        assert set(c for c, _ in changes(index, v2).values()) == {"added"}
        index.index_commit(v1)
        assert_linked(index, v1, v2)


def test_version_store_lineage_and_symbol_diff(repo):
    path, v1, v2 = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        index.index_commit(v1)
        index.index_commit(v2)
        store = IndexVersionStore(index)
        assert [v.id for v in store.versions()] == [v1, v2]
        lineage_id = changes(index, v2)["read_config"][1]
        nodes = store.lineage(lineage_id)
        assert [(n.symbol_id, n.change_type) for n in nodes] == [
            ("cfg.py::load_config", ChangeType.ADDED), ("cfg.py::read_config", ChangeType.RENAMED),
        ]
        assert nodes[1].commit_message.startswith("rename load_config")
        diff = store.diff_symbol("read_config", v1, v2)
        assert diff.change_type == ChangeType.RENAMED and "def load_config" in diff.old_text
        assert '+    data.setdefault("debug", False)' in diff.unified_diff
        assert store.diff_symbol("text/util.py::slugify", v1, v2).change_type == ChangeType.MOVED
        assert store.diff_symbol("brand_new", v1, v2).change_type == ChangeType.ADDED


def test_history_search_reports_each_hits_lineage(repo):
    path, v1, v2 = repo
    with VersionedIndex(path, path / "idx.sqlite") as index:
        index.index_commit(v1)
        out = search_index(index, "config json validate", v2, 5, include_history=True, cfg=SeraphConfig())
    hits = [r for r in out["results"] if r["symbol"] in ("load_config", "read_config")]
    assert len({r["lineage_id"] for r in hits}) == 1
    assert [h["change_type"] for h in hits[0]["history"]] == ["added", "renamed"]


def test_small_bodies_only_link_by_identical_content():
    class C:
        def __init__(self, path, symbol, text):
            self.path, self.symbol, self.text, self.start_line = path, symbol, text, 1
            self.content_hash, self.language = str(hash(text)), "python"

    old = [C("a.py", "legacy", "def legacy():\n    return 42\n")]
    new = [C("a.py", "brand_new", "def brand_new():\n    return 'fresh'\n")]
    assert sorted(m.change_type for m in match_symbols(old, new)) == [ChangeType.ADDED, ChangeType.DELETED]
