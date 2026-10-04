"""`seraph smoke`: a self-contained check of indexing, version search, history and compare."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from .config import SeraphConfig
from .index import VersionedIndex
from .service import compare_versions, search_index

V1 = {
    "config.py": "import json\n\n\ndef parse_config(path):\n    return json.load(open(path))\n",
    "net.py": "import socket\n\n\ndef open_socket(host, port):\n    return socket.create_connection((host, port))\n",
    "settings.ts": "export function loadSettings(path: string) {\n  return readJson(path);\n}\n",
}
V2 = {
    "config.py": (
        "import yaml\n\n\ndef parse_config(path):\n    data = yaml.safe_load(open(path))\n"
        "    validate_schema(data)\n    return data\n"
    ),
    "cache.py": "def cache_result(key, value, store):\n    store[key] = value\n    return value\n",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        (repo / name).write_text(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def run() -> int:
    cfg = SeraphConfig()
    checks: list[tuple[str, bool, float, str]] = []

    def check(name: str, fn) -> None:
        start = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append((name, ok, (time.perf_counter() - start) * 1000, detail))

    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "smoke@seraph.dev")
        _git(repo, "config", "user.name", "Seraph Smoke")
        v1 = _commit(repo, V1, "initial: json config, sockets, settings")
        v2 = _commit(repo, V2, "switch config to yaml with schema validation, add cache")

        with VersionedIndex(repo) as index:
            def first_index():
                s = index.index_commit(v1)
                return s.parsed_files == 3, f"parsed {s.parsed_files} files, {s.total_chunks} chunks"

            def incremental_index():
                s = index.index_commit(v2)
                return s.parsed_files == 2 and s.reused_chunks >= 2, (
                    f"parsed {s.parsed_files} changed files, reused {s.reused_chunks}/{s.total_chunks} chunks"
                )

            def top(query: str, ref: str, history: bool = False):
                return search_index(index, query, ref, 3, history, cfg=cfg)["results"]

            def search_new():
                hit = top("parse the config file as yaml and validate the schema", v2)[0]
                return hit["symbol"] == "parse_config" and "yaml" in hit["text"], f"#1 {hit['symbol']} @{v2[:7]}"

            def search_old():
                hit = top("parse the config file", v1)[0]
                return hit["symbol"] == "parse_config" and "json" in hit["text"], f"#1 {hit['symbol']} (json version) @{v1[:7]}"

            def search_ts():
                hit = top("load settings from a path", v2)[0]
                return hit["symbol"] == "loadSettings", f"#1 {hit['symbol']} in {hit['path']}"

            def history():
                hits = [h for h in top("parse config", v2, history=True) if h["symbol"] == "parse_config"]
                commits = {h["commit"] for h in hits}
                return commits == {v1, v2}, f"parse_config found in {len(commits)} versions"

            def compare():
                out = compare_versions(index, v1, v2, 50)
                changes = {(c["change_type"], c["symbol"].split("::")[-1]) for c in out["symbols"]}
                ok = ("added", "cache_result") in changes and ("modified", "parse_config") in changes
                return ok, f"added {out['summary']['added']}, modified {out['summary']['modified']}"

            check("index first commit", first_index)
            check("index next commit (incremental)", incremental_index)
            check("search new version", search_new)
            check("search old version", search_old)
            check("TypeScript symbol search", search_ts)
            check("search all versions", history)
            check("compare versions", compare)

    width = max(len(name) for name, *_ in checks)
    for name, ok, ms, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {ms:7.1f} ms  {detail}")
    passed = sum(ok for _, ok, *_ in checks)
    print(f"\n{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1
