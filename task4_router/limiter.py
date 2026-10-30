"""Token-aware sliding-window rate limiter backed by an on-disk SQLite file.

Model: every admitted request inserts one row (key_hash, ts_ms, tokens). A tenant's usage is
SUM(tokens) over rows with ts in (now - window, now]. Rows older than the window are evicted inside
the same transaction, so the table never grows past one window of traffic.

Atomicity: check + insert happen under `BEGIN IMMEDIATE` (SQLite write lock), so concurrent requests
- in this process, other threads, or other worker processes - cannot both slip into the last slot.

Token awareness: callers reserve an *estimate* up front, then `settle()` replaces it with the
provider-reported actual (or `refund()` deletes it if no completion was produced).
Raw API keys are never stored; only a SHA-256 digest.
"""
from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable

DEFAULT_LIMIT = 50_000
DEFAULT_WINDOW_MS = 60_000


@dataclass(frozen=True)
class Decision:
    allowed: bool
    used: int = 0              # tokens in window after this decision (incl. reservation if allowed)
    retry_after_ms: int = 0    # when denied: earliest time enough budget frees up
    reservation_id: int | None = None
    too_large: bool = False    # request alone exceeds the limit: retrying can never help


def key_hash(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class SlidingWindowLimiter:
    def __init__(self, db_path: str, limit: int = DEFAULT_LIMIT, window_ms: int = DEFAULT_WINDOW_MS,
                 clock_ms: Callable[[], int] | None = None) -> None:
        self.db_path, self.limit, self.window_ms = db_path, limit, window_ms
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")  # persistent; readers don't block the writer
            c.execute("CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, ts INTEGER NOT NULL, tokens INTEGER NOT NULL)")
            c.execute("CREATE INDEX IF NOT EXISTS usage_key_ts ON usage(key, ts)")
            c.execute("CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts)")

    def _conn(self) -> sqlite3.Connection:
        # autocommit mode: we issue BEGIN/COMMIT explicitly. busy timeout absorbs writer contention.
        c = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    # ---- sync core (run in a worker thread so the event loop never blocks on disk) ----
    def _reserve(self, api_key: str, tokens: int) -> Decision:
        if tokens > self.limit:
            return Decision(False, too_large=True)
        kh, now = key_hash(api_key), self.clock_ms()
        c = self._conn()
        try:
            c.execute("BEGIN IMMEDIATE")
            c.execute("DELETE FROM usage WHERE ts <= ?", (now - self.window_ms,))  # eviction (all tenants)
            rows = c.execute("SELECT ts, tokens FROM usage WHERE key=? ORDER BY ts", (kh,)).fetchall()
            used = sum(t for _, t in rows)
            if used + tokens > self.limit:
                need, freed, retry = used + tokens - self.limit, 0, self.window_ms
                for ts, tk in rows:
                    freed += tk
                    if freed >= need:  # oldest rows must expire until enough is freed
                        retry = ts + self.window_ms - now
                        break
                c.execute("COMMIT")
                return Decision(False, used=used, retry_after_ms=max(retry, 1))
            rid = c.execute("INSERT INTO usage(key, ts, tokens) VALUES (?,?,?)", (kh, now, tokens)).lastrowid
            c.execute("COMMIT")
            return Decision(True, used=used + tokens, reservation_id=rid)
        except BaseException:
            if c.in_transaction:
                c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    def _exec(self, sql: str, args: tuple) -> None:
        c = self._conn()
        try:
            c.execute(sql, args)
        finally:
            c.close()

    def _usage(self, api_key: str) -> int:
        c = self._conn()
        try:
            now = self.clock_ms()
            return c.execute("SELECT COALESCE(SUM(tokens),0) FROM usage WHERE key=? AND ts > ?", (key_hash(api_key), now - self.window_ms)).fetchone()[0]
        finally:
            c.close()

    # ---- async API ----
    async def reserve(self, api_key: str, tokens: int) -> Decision:
        return await asyncio.to_thread(self._reserve, api_key, max(tokens, 0))

    async def settle(self, reservation_id: int, actual_tokens: int) -> None:
        await asyncio.to_thread(self._exec, "UPDATE usage SET tokens=? WHERE id=?", (max(actual_tokens, 0), reservation_id))

    async def refund(self, reservation_id: int) -> None:
        await asyncio.to_thread(self._exec, "DELETE FROM usage WHERE id=?", (reservation_id,))

    async def usage(self, api_key: str) -> int:
        return await asyncio.to_thread(self._usage, api_key)
