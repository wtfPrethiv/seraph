import subprocess
from pathlib import Path

from seraph.index import VersionedIndex
from seraph.service import compare_versions


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


def commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        (repo / name).write_text(text)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def test_compare_versions_reports_symbols_dependencies_and_commits(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Seraph Test")
    v1 = commit(tmp_path, {
        "cfg.py": LOAD + "def helper(x):\n    return x + 1\n",
        "app.py": "from cfg import load_config\n\n\ndef main():\n    return load_config('a')\n",
    }, "initial")
    commit(tmp_path, {"cfg.py": READ + "def helper(x):\n    return x + 1\n",
                      "app.py": "from cfg import read_config\n\n\ndef main():\n    return read_config('a')\n"},
           "rename load_config")
    v3 = commit(tmp_path, {
        "cfg.py": READ + "def helper(x):\n    return x + 2\n",
        "app.py": "from cfg import helper, read_config\n\n\ndef main():\n    return helper(read_config('a'))\n",
    }, "main post-processes the config with helper")

    with VersionedIndex(tmp_path, tmp_path / "idx.sqlite") as index:
        out = compare_versions(index, v1, v3)

    assert (out["from"], out["to"]) == (v1, v3)
    assert out["summary"] == {"added": 0, "modified": 2, "renamed": 1, "moved": 0, "deleted": 0}
    by_symbol = {s["symbol"]: s for s in out["symbols"]}
    assert set(by_symbol) == {"app.py::main", "cfg.py::helper", "cfg.py::read_config"}
    renamed = by_symbol["cfg.py::read_config"]
    assert renamed["change_type"] == "renamed" and renamed["previous"] == "cfg.py::load_config"
    assert "-def load_config(path):" in renamed["diff"] and "+def read_config(path):" in renamed["diff"]

    added = {(e["src"], e["dst"], e["kind"]) for e in out["dependencies"]["added"]}
    assert ("app.py::main", "cfg.py::helper", "calls") in added
    assert ("app.py", "cfg.py::helper", "imports") in added
    # The rename alone does not show up as a removed and re-added call.
    assert ("app.py::main", "cfg.py::read_config", "calls") not in added
    assert out["dependencies"]["removed"] == []

    assert [c["subject"] for c in out["commits"]] == ["main post-processes the config with helper", "rename load_config"]
    assert set(out["commits"][0]["files"]) == {"app.py", "cfg.py"}
    assert "cfg.py::helper" in out["commits"][0]["symbols"]


def test_compare_identical_versions_is_empty(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Seraph Test")
    v1 = commit(tmp_path, {"a.py": "def f():\n    return 1\n"}, "initial")
    with VersionedIndex(tmp_path, tmp_path / "idx.sqlite") as index:
        out = compare_versions(index, v1, "HEAD")
    assert out["symbols"] == [] and out["commits"] == [] and not any(out["summary"].values())
