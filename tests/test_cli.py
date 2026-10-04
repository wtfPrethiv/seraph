import subprocess
from pathlib import Path

from seraph.cli import main


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def test_cli_symbol_deps_and_compare(tmp_path, capsys, monkeypatch):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Seraph Test")
    load = (
        "def load_config(path):\n    with open(path) as handle:\n        data = json.load(handle)\n"
        "    validate(data)\n    return data\n"
    )
    read = (
        "def read_config(path):\n    with open(path) as handle:\n        data = json.load(handle)\n"
        "    validate(data)\n    return data\n"
    )
    (tmp_path / "cfg.py").write_text(load)
    (tmp_path / "app.py").write_text("from cfg import load_config\n\n\ndef main():\n    return load_config('a')\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    v1 = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "cfg.py").write_text(read)
    (tmp_path / "app.py").write_text("from cfg import read_config\n\n\ndef main():\n    return read_config('a')\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "rename")
    db = tmp_path / "idx.sqlite"
    common = ["seraph", "--repo", str(tmp_path), "--db", str(db)]

    monkeypatch.setattr("sys.argv", [*common, "symbol", "read_config"])
    main()
    assert "cfg.py::read_config" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", [*common, "deps", "main", "--direction", "out"])
    main()
    assert "app.py::main -> cfg.py::read_config" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", [*common, "compare", v1, "HEAD"])
    main()
    out = capsys.readouterr().out
    assert "renamed" in out and "cfg.py::read_config" in out
