"""
RQ (Redis Queue) wrapper for running agent jobs out-of-process, instead of
the unbounded threading.Thread-per-request dispatch: gives a real
concurrency cap (N workers = N concurrent runs) and jobs survive an
API-process restart. Opt-in via $AGENT_QUEUE_ENABLED; api/server.py falls
back to threading.Thread when it's off or Redis isn't reachable.

Run a worker (separate process, same codebase/venv):
    AGENT_QUEUE_ENABLED=1 REDIS_URL=redis://localhost:6379/0 rq worker agent-runs
"""

from __future__ import annotations

import os
from typing import Any, Optional

DEFAULT_QUEUE_NAME = "agent-runs"
DEFAULT_JOB_TIMEOUT = "20m"  # generous vs AgentBudget.deadline_seconds=600s default + workspace setup

_queue = None


def is_queue_enabled() -> bool:
    return os.environ.get("AGENT_QUEUE_ENABLED", "0").strip().lower() in {"1", "true", "yes"}


def get_queue():
    """
    Returns an rq.Queue, or None if Redis isn't configured/reachable.

    Deliberately does NOT reuse db.cache's shared connection: that one is
    created with decode_responses=True (convenient for cache.py's own string
    keys/JSON pub/sub payloads), but RQ stores job data as pickled binary
    blobs in Redis hashes -- handing it a connection that auto-UTF-8-decodes
    every response corrupts that binary data (Job.fetch() raises
    UnicodeDecodeError trying to decode raw pickle bytes as text). RQ needs
    its own raw-bytes connection to the same Redis instance.
    """
    global _queue
    if _queue is not None:
        return _queue

    from . import cache
    if not cache.is_configured():
        return None

    try:
        import redis as redis_lib
        from rq import Queue
        conn = redis_lib.from_url(os.environ["REDIS_URL"])  # no decode_responses -- RQ needs raw bytes
        conn.ping()
        _queue = Queue(DEFAULT_QUEUE_NAME, connection=conn)
        return _queue
    except Exception as e:
        print(f"[!] Could not create RQ queue: {e}")
        return None


def enqueue_run(job_fn: Any, *args, job_id: Optional[str] = None, timeout: str = DEFAULT_JOB_TIMEOUT, **kwargs):
    """
    Enqueue job_fn(*args, **kwargs) to run in a worker process. job_id lets
    the caller use its own run_id as the RQ job id (so job status can be
    looked up later without keeping a separate mapping).

    Raises if the queue isn't available -- callers should check
    is_queue_enabled() + catch this and fall back to threading.Thread
    themselves (see api/server.py's _dispatch_agent_run).
    """
    q = get_queue()
    if q is None:
        raise RuntimeError("RQ queue unavailable (Redis not configured/reachable)")
    return q.enqueue(job_fn, *args, job_id=job_id, job_timeout=timeout, **kwargs)


def fetch_job(job_id: str):
    """Returns the rq.job.Job for job_id, or None if it doesn't exist / queue unavailable."""
    q = get_queue()
    if q is None:
        return None
    try:
        from rq.job import Job
        return Job.fetch(job_id, connection=q.connection)  # same raw-bytes connection as get_queue()
    except Exception:
        return None
