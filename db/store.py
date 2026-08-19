"""
Postgres persistence for sessions/messages/runs (see db/schema.sql).

Optional and best-effort: api/server.py's in-memory _sessions/_runs dicts
stay the source of truth for the live request/SSE hot path; every function
here is an additional write-through for durability that never raises -- a
DB hiccup means "this write didn't persist", not "the request failed".
Inactive unless $DATABASE_URL is set and reachable.

Uses psycopg3's sync ConnectionPool, not asyncpg: agent runs execute in
background threads (not asyncio tasks), so one sync client reachable from
async routes via asyncio.to_thread is simpler than two DB client types for
what's a handful of small, infrequent metadata writes.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

_pool = None


def is_configured() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def available() -> bool:
    return _pool is not None


def init_pool(dsn: Optional[str] = None, timeout: float = 5.0) -> bool:
    """Create the connection pool if a DSN is given/configured and reachable. Returns True if usable."""
    global _pool
    if _pool is not None:
        return True

    dsn = dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        return False

    try:
        from psycopg_pool import ConnectionPool
        pool = ConnectionPool(dsn, min_size=1, max_size=5, open=True, timeout=timeout)
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        _pool = pool
        return True
    except Exception as e:
        print(f"[!] Postgres unavailable, staying on in-memory store: {e}")
        _pool = None
        return False


def close_pool() -> None:
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
        _pool = None


@contextmanager
def _conn():
    with _pool.connection() as conn:
        yield conn


def _safe(fn_name: str, fn, *args, **kwargs):
    if not available():
        return None
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[!] db.store.{fn_name} failed (continuing without persistence): {e}")
        return None


# ── sessions ──────────────────────────────────────────────────────────────

def create_session(session_id: str, repo_label: str, repo_root: str) -> None:
    def _do():
        with _conn() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, repo_label, repo_root) VALUES (%s, %s, %s) "
                "ON CONFLICT (session_id) DO NOTHING",
                (session_id, repo_label, repo_root),
            )
    _safe("create_session", _do)


def touch_session(session_id: str) -> None:
    def _do():
        with _conn() as conn:
            conn.execute("UPDATE sessions SET last_active = now() WHERE session_id = %s", (session_id,))
    _safe("touch_session", _do)


def close_session(session_id: str) -> None:
    def _do():
        with _conn() as conn:
            conn.execute(
                "UPDATE sessions SET closed_at = now() WHERE session_id = %s AND closed_at IS NULL",
                (session_id,),
            )
    _safe("close_session", _do)


def list_sessions(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    def _do():
        with _conn() as conn:
            rows = conn.execute(
                "SELECT session_id, repo_label, repo_root, created_at, last_active, closed_at "
                "FROM sessions ORDER BY last_active DESC LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
            cols = ["session_id", "repo_label", "repo_root", "created_at", "last_active", "closed_at"]
            return [dict(zip(cols, r)) for r in rows]
    return _safe("list_sessions", _do) or []


# ── messages ──────────────────────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str, run_id: Optional[str] = None) -> None:
    def _do():
        with _conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, run_id) VALUES (%s, %s, %s, %s)",
                (session_id, role, content, run_id),
            )
    _safe("add_message", _do)


def get_messages(session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    def _do():
        with _conn() as conn:
            rows = conn.execute(
                "SELECT role, content, run_id, created_at FROM messages "
                "WHERE session_id = %s ORDER BY created_at ASC LIMIT %s",
                (session_id, limit),
            ).fetchall()
            cols = ["role", "content", "run_id", "created_at"]
            return [dict(zip(cols, r)) for r in rows]
    return _safe("get_messages", _do) or []


# ── runs ──────────────────────────────────────────────────────────────────

def create_run(run_id: str, session_id: Optional[str], task: str, repo_url: Optional[str] = None) -> None:
    def _do():
        with _conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, session_id, task, repo_url, run_status) "
                "VALUES (%s, %s, %s, %s, 'running') ON CONFLICT (run_id) DO NOTHING",
                (run_id, session_id, task, repo_url),
            )
    _safe("create_run", _do)


def finish_run(run_id: str, result: Dict[str, Any]) -> None:
    def _do():
        with _conn() as conn:
            conn.execute(
                """
                UPDATE runs SET
                    run_status = %s, validation = %s, stop_reason = %s,
                    replan_count = %s, tool_call_count = %s,
                    files_modified = %s, final_answer = %s,
                    result = %s, finished_at = now()
                WHERE run_id = %s
                """,
                (
                    result.get("run_status"),
                    result.get("validation"),
                    result.get("stop_reason"),
                    result.get("replan_count"),
                    result.get("tool_call_count"),
                    json.dumps(result.get("files_modified", [])),
                    result.get("final_answer"),
                    json.dumps(result, default=str),
                    run_id,
                ),
            )
    _safe("finish_run", _do)


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    def _do():
        with _conn() as conn:
            row = conn.execute(
                "SELECT run_id, session_id, task, repo_url, run_status, validation, stop_reason, "
                "replan_count, tool_call_count, files_modified, final_answer, result, created_at, finished_at "
                "FROM runs WHERE run_id = %s",
                (run_id,),
            ).fetchone()
            if not row:
                return None
            cols = [
                "run_id", "session_id", "task", "repo_url", "run_status", "validation", "stop_reason",
                "replan_count", "tool_call_count", "files_modified", "final_answer", "result",
                "created_at", "finished_at",
            ]
            return dict(zip(cols, row))
    return _safe("get_run", _do)


def list_runs(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    def _do():
        with _conn() as conn:
            rows = conn.execute(
                "SELECT run_id, task, run_status, created_at, finished_at FROM runs "
                "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
            cols = ["run_id", "task", "run_status", "created_at", "finished_at"]
            return [dict(zip(cols, r)) for r in rows]
    return _safe("list_runs", _do) or []


def delete_run(run_id: str) -> bool:
    def _do():
        with _conn() as conn:
            cur = conn.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
            return cur.rowcount > 0
    return bool(_safe("delete_run", _do))
