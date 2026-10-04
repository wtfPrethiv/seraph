"""Shared plumbing for API backends: rate limiting, retries, persistent per-item cache."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

_RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE", "timed out")


class MissingAPIKeyError(RuntimeError):
    pass


def require_key(env: str) -> str:
    key = os.environ.get(env)
    if not key:
        raise MissingAPIKeyError(
            f"{env} is not set. Set it (PowerShell: setx {env} <key>, then reopen the terminal) "
            "or switch this component to a local model in the config."
        )
    return key


class RateLimiter:
    """Spaces calls to at most `rpm` per minute."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval


class QuotaExhaustedError(RuntimeError):
    """A daily quota is used up; retrying today will not help. Switch keys or wait."""


_RETRY_DELAY = re.compile(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", re.I)


def with_retries[T](fn: Callable[[], T], attempts: int = 8, base_delay: float = 4.0) -> T:
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg:
                raise QuotaExhaustedError(f"daily API quota exhausted: {msg[:300]}") from e
            if i == attempts - 1 or not any(code in msg for code in _RETRYABLE):
                raise
            hint = _RETRY_DELAY.search(msg)
            delay = float(hint.group(1)) + 1 if hint else min(base_delay * 2**i, 120.0)
            log.warning("API call failed (%s); retrying in %.0fs", msg[:120], delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


class KVCache:
    """SQLite key -> bytes store; survives interrupted runs (free-tier daily limits)."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB)")
        self._db.commit()

    @staticmethod
    def key(*parts: str) -> str:
        h = hashlib.sha1()
        for p in parts:
            h.update(p.encode())
            h.update(b"\x00")
        return h.hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(chunk))})"
            out.update(self._db.execute(q, chunk).fetchall())
        return out

    def put_many(self, items: dict[str, bytes]) -> None:
        self._db.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)", items.items())
        self._db.commit()
