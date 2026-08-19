from __future__ import annotations

"""
Redis wiring: a shared connection, session-activity TTL (replaces the
poll-every-300s watchdog thread), and pub/sub for SSE run events.

The pub/sub layer exists specifically for db/queue.py's RQ integration: once
an agent run executes in a separate `rq worker` OS process instead of a
background thread inside the FastAPI process, a plain in-process
queue.Queue can't carry its progress events across that process boundary --
Redis pub/sub can. api/server.py's emit() publishes here when Redis is
configured; a small relay thread in the FastAPI process re-forwards those
messages into the existing local queue.Queue that SSE streaming already
polls, so stream_run()'s code doesn't need to know or care whether the run
is executing in-process or in a worker.

Optional and best-effort, same shape as db/store.py: everything degrades to
a no-op / None if $REDIS_URL isn't set or unreachable, so the app keeps
working exactly as before (thread-based watchdog, in-process queue) with
zero configuration.
"""

import json
import os
from typing import Any, Dict, Iterator, Optional

_redis = None

SESSION_TTL_SECONDS = 1800  # 30 min idle -> expired; matches the prior watchdog's threshold


def is_configured() -> bool:
    return bool(os.environ.get("REDIS_URL"))


def available() -> bool:
    return _redis is not None


def init_redis(url: Optional[str] = None, timeout: float = 5.0) -> bool:
    global _redis
    if _redis is not None:
        return True

    url = url or os.environ.get("REDIS_URL")
    if not url:
        return False

    try:
        import redis as redis_lib
        client = redis_lib.from_url(
            url, socket_connect_timeout=timeout, socket_timeout=timeout, decode_responses=True
        )
        client.ping()
        _redis = client
        return True
    except Exception as e:
        print(f"[!] Redis unavailable, staying on thread/in-memory fallbacks: {e}")
        _redis = None
        return False


def close_redis() -> None:
    global _redis
    if _redis is not None:
        try:
            _redis.close()
        except Exception:
            pass
        _redis = None


def get_connection():
    """Raw redis-py client, e.g. for db/queue.py's RQ Queue(connection=...)."""
    return _redis


# ── session activity TTL ────────────────────────────────────────────────────

def touch_session_ttl(session_id: str, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
    if not available():
        return
    try:
        _redis.set(f"session_active:{session_id}", "1", ex=ttl_seconds)
    except Exception:
        pass


def is_session_active(session_id: str) -> Optional[bool]:
    """None (not True/False) when Redis is unavailable, so callers know to fall back to their own tracking."""
    if not available():
        return None
    try:
        return bool(_redis.exists(f"session_active:{session_id}"))
    except Exception:
        return None


def clear_session_ttl(session_id: str) -> None:
    if not available():
        return
    try:
        _redis.delete(f"session_active:{session_id}")
    except Exception:
        pass


# ── SSE pub/sub ──────────────────────────────────────────────────────────

def publish_event(run_id: str, event: Dict[str, Any]) -> None:
    if not available():
        return
    try:
        _redis.publish(f"run_events:{run_id}", json.dumps(event, default=str))
    except Exception:
        pass


def open_subscription(run_id: str, timeout: float = 5.0):
    """
    Subscribe to run_id's event channel and block until Redis has actually
    confirmed the subscription, returning the raw pubsub handle (or None if
    Redis isn't available).

    Call this BEFORE doing anything that might publish to the channel (e.g.
    enqueuing the job whose events you're about to consume). Redis pub/sub
    has no backlog/replay -- a publish that happens before a subscriber's
    SUBSCRIBE has been acked by the server is simply never delivered to that
    subscriber. `pubsub.subscribe(...)` alone only sends the command; it
    doesn't wait for the server's ack, so a publisher racing right behind it
    (e.g. an RQ worker that dequeues and finishes a fast job in milliseconds)
    can complete -- including publishing the terminal "run_result" event --
    before the subscription is actually live. Whoever's waiting on that
    event then blocks forever with no error, which is a much worse failure
    mode than a normal exception. Consuming one `get_message()` here forces
    that ack (redis-py's documented way to synchronize on it) before this
    function returns, so a publish issued by the caller right after this
    call is guaranteed to be seen. Pair with iter_subscription() to consume it.
    """
    if not available():
        return None
    pubsub = _redis.pubsub()
    pubsub.subscribe(f"run_events:{run_id}")
    pubsub.get_message(timeout=timeout)  # blocks for the SUBSCRIBE ack; see docstring
    return pubsub


def iter_subscription(pubsub) -> Iterator[Dict[str, Any]]:
    """
    Consume an already-subscribed pubsub handle (see open_subscription()).
    Yields decoded event dicts as they're published, and stops (returns)
    right after an event of type "run_result" -- api/server.py's
    _execute_agent_run() always publishes exactly one of these, from its
    `finally` block, as the last thing it does, regardless of whether the run
    succeeded, failed, or hit a budget limit. Callers don't need a separate
    sentinel value; the generator ending IS the signal. Closes `pubsub` on
    the way out either way (normal completion or the caller breaking early).

    Known gap: if the worker process is killed hard enough that even the
    `finally` block never runs (not a Python exception, an actual process
    kill), this blocks forever with no timeout. Not solved here -- same
    class of problem as the "crashed run stays stuck in Postgres" gap noted
    in db/cleanup.sql; deferred for the same reason (no immediate impact).
    """
    try:
        for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            try:
                event = json.loads(message["data"])
            except (TypeError, ValueError):
                continue
            yield event
            if event.get("type") == "run_result":
                return
    finally:
        try:
            pubsub.close()
        except Exception:
            pass


def subscribe_events(run_id: str) -> Iterator[Dict[str, Any]]:
    """
    Convenience wrapper: subscribe-then-consume in one call. Blocking
    generator -- call from a dedicated thread, not the asyncio event loop.

    Only safe to use when nothing can publish to this channel until after
    the caller starts iterating (e.g. tests that subscribe first and publish
    from a separate thread afterwards). If you're about to trigger the
    publisher yourself (like enqueuing the job), call open_subscription()
    first and enqueue only after it returns -- see that function's docstring
    for why, and _dispatch_agent_run() in api/server.py for the real caller
    that needs this ordering.
    """
    pubsub = open_subscription(run_id)
    if pubsub is None:
        return
    yield from iter_subscription(pubsub)
