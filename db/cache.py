"""
Redis wiring: a shared connection, session-activity TTL (replaces the old
poll-every-300s watchdog thread), and pub/sub for SSE run events.

Pub/sub exists for db/queue.py's RQ integration: an agent run in a separate
`rq worker` process can't share the FastAPI process's in-memory
queue.Queue, so it publishes progress here instead, and a relay thread
forwards those messages into the local queue.Queue that SSE streaming
already polls -- stream_run() doesn't need to know whether the run is
in-process or in a worker.

Optional and best-effort, same shape as db/store.py: degrades to a no-op /
None if $REDIS_URL isn't set or unreachable, so the app keeps working with
zero configuration.
"""

from __future__ import annotations

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
    Subscribe to run_id's event channel and block until Redis confirms the
    subscription. Returns the raw pubsub handle, or None if unavailable.

    Call this BEFORE anything that might publish to the channel (e.g.
    enqueuing the job you're about to consume events from). Redis pub/sub
    has no backlog -- a publish issued before the SUBSCRIBE is acked is
    dropped, and a fast worker can publish everything (including the
    terminal run_result) before a naive subscribe-then-listen catches up,
    hanging the reader forever with no error. Consuming one get_message()
    here forces that ack first. Pair with iter_subscription() to consume.
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
    Yields events as published, stops right after a "run_result" event
    (api/server.py's _execute_agent_run() always publishes exactly one, from
    its `finally` block, win or lose), and closes `pubsub` either way.

    Known gap: if the worker is killed hard enough that even `finally`
    never runs, this blocks forever with no timeout -- same class of issue
    as db/cleanup.sql's "crashed run stays stuck" gap, deferred for the
    same reason.
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

    Only safe when nothing can publish on this channel until the caller
    starts iterating. If you're about to trigger the publisher yourself
    (e.g. enqueuing a job), call open_subscription() first instead and
    enqueue only after it returns -- see its docstring for why.
    """
    pubsub = open_subscription(run_id)
    if pubsub is None:
        return
    yield from iter_subscription(pubsub)
