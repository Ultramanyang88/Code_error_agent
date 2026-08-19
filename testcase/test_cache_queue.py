"""
Tests for db/cache.py (Redis: session TTL + SSE pub/sub) and db/queue.py (RQ).

Same split as test_db_store.py/test_sandbox.py: test_all.py's contract is
"no external services required". These need a reachable Redis:

    docker compose up -d redis
    REDIS_URL=redis://localhost:6379/0 python -m pytest testcase/test_cache_queue.py -v

Skipped automatically (not failed) when $REDIS_URL isn't set, so this file
is still safe in a default `pytest testcase/` sweep -- only the
"unconfigured is a safe no-op" tests run in that case.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REDIS_URL = os.environ.get("REDIS_URL")


def _sample_job_for_test(x):
    """Module-level on purpose: RQ pickles jobs by (module, qualname) reference,
    so a function nested inside a test method can't be enqueued -- it has no
    importable path for a worker to resolve it back from."""
    return x * 2


@unittest.skipUnless(REDIS_URL, "REDIS_URL not set -- see module docstring to run this class")
class TestCacheLive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from db import cache
        cls.cache = cache
        if not cache.init_redis(REDIS_URL, timeout=5.0):
            raise unittest.SkipTest("Redis not reachable at $REDIS_URL")

    @classmethod
    def tearDownClass(cls):
        cls.cache.close_redis()

    def setUp(self):
        self.session_id = f"test-{uuid.uuid4().hex[:8]}"
        self.run_id = f"run-{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        self.cache.clear_session_ttl(self.session_id)

    def test_session_ttl_round_trip(self):
        # False (not None) for a key that just doesn't exist -- None is
        # reserved for "Redis itself is unavailable", covered separately in
        # TestCacheUnconfigured.test_is_session_active_returns_none_not_false.
        self.assertFalse(self.cache.is_session_active("never-touched-" + self.session_id))
        self.cache.touch_session_ttl(self.session_id, ttl_seconds=60)
        self.assertTrue(self.cache.is_session_active(self.session_id))

    def test_clear_session_ttl(self):
        self.cache.touch_session_ttl(self.session_id, ttl_seconds=60)
        self.assertTrue(self.cache.is_session_active(self.session_id))
        self.cache.clear_session_ttl(self.session_id)
        self.assertFalse(self.cache.is_session_active(self.session_id))

    def test_session_ttl_actually_expires(self):
        self.cache.touch_session_ttl(self.session_id, ttl_seconds=1)
        self.assertTrue(self.cache.is_session_active(self.session_id))
        time.sleep(1.5)
        self.assertFalse(self.cache.is_session_active(self.session_id))

    def test_publish_and_subscribe_round_trip(self):
        import threading
        received = []

        def _listen():
            for event in self.cache.subscribe_events(self.run_id):
                received.append(event)

        t = threading.Thread(target=_listen, daemon=True)
        t.start()
        time.sleep(0.3)  # let the subscription actually establish before publishing

        self.cache.publish_event(self.run_id, {"type": "step_start", "data": {"step_id": 1}})
        self.cache.publish_event(self.run_id, {"type": "run_result", "data": {"run_status": "completed"}})
        t.join(timeout=5)

        self.assertFalse(t.is_alive(), "subscriber should stop right after run_result")
        types = [e["type"] for e in received]
        self.assertEqual(types, ["step_start", "run_result"])

    def test_subscribe_stops_after_run_result_even_with_more_messages(self):
        import threading
        received = []

        def _listen():
            for event in self.cache.subscribe_events(self.run_id):
                received.append(event)

        t = threading.Thread(target=_listen, daemon=True)
        t.start()
        time.sleep(0.3)

        self.cache.publish_event(self.run_id, {"type": "run_result", "data": {}})
        self.cache.publish_event(self.run_id, {"type": "step_start", "data": {}})  # published too late, must be ignored
        t.join(timeout=5)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["type"], "run_result")

    def test_open_subscription_blocks_until_ready_no_sleep_needed(self):
        # Regression test for the race that api/server.py's _run_via_queue()
        # used to have: subscribe_events() alone only *sends* the SUBSCRIBE
        # command, it doesn't wait for Redis's ack, so a publish issued
        # immediately after starting a subscribe_events() listener thread
        # (no time.sleep to let it "settle", unlike every other test in this
        # class) can race ahead of the subscription and be dropped silently.
        # open_subscription() is supposed to close that window by blocking
        # until the subscription is actually confirmed -- prove it by
        # publishing with zero delay and confirming nothing is lost, across
        # several iterations to make the race window's absence more than luck.
        for _ in range(20):
            run_id = f"run-{uuid.uuid4().hex[:8]}"
            pubsub = self.cache.open_subscription(run_id)
            self.assertIsNotNone(pubsub)

            received = []
            import threading

            def _listen():
                for event in self.cache.iter_subscription(pubsub):
                    received.append(event)

            t = threading.Thread(target=_listen, daemon=True)
            t.start()

            # No sleep here on purpose -- open_subscription() already
            # blocked until the SUBSCRIBE was acked, so this publish is
            # guaranteed to be seen even issued right away.
            self.cache.publish_event(run_id, {"type": "run_result", "data": {"ok": True}})
            t.join(timeout=5)

            self.assertFalse(t.is_alive(), "subscriber should have stopped after run_result")
            self.assertEqual(len(received), 1, "run_result was dropped -- subscription wasn't actually ready")
            self.assertEqual(received[0]["type"], "run_result")


@unittest.skipIf(REDIS_URL, "this class specifically covers the NO-REDIS fallback path")
class TestCacheUnconfigured(unittest.TestCase):

    def test_not_available_without_url(self):
        from db import cache
        self.assertFalse(cache.available())

    def test_writes_do_not_raise(self):
        from db import cache
        cache.touch_session_ttl("x")
        cache.clear_session_ttl("x")
        cache.publish_event("x", {"type": "done", "data": {}})

    def test_is_session_active_returns_none_not_false(self):
        # None (not False) signals "can't tell, Redis is unavailable" so
        # callers know to fall back to their own tracking instead of
        # concluding "the session is dead".
        from db import cache
        self.assertIsNone(cache.is_session_active("x"))

    def test_subscribe_events_yields_nothing(self):
        from db import cache
        self.assertEqual(list(cache.subscribe_events("x")), [])


@unittest.skipUnless(REDIS_URL, "REDIS_URL not set -- see module docstring to run this class")
class TestQueueLive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from db import cache
        if not cache.init_redis(REDIS_URL, timeout=5.0):
            raise unittest.SkipTest("Redis not reachable at $REDIS_URL")

    def test_is_queue_enabled_env_var(self):
        from db.queue import is_queue_enabled
        old = os.environ.get("AGENT_QUEUE_ENABLED")
        try:
            os.environ["AGENT_QUEUE_ENABLED"] = "1"
            self.assertTrue(is_queue_enabled())
            os.environ["AGENT_QUEUE_ENABLED"] = "0"
            self.assertFalse(is_queue_enabled())
        finally:
            if old is None:
                os.environ.pop("AGENT_QUEUE_ENABLED", None)
            else:
                os.environ["AGENT_QUEUE_ENABLED"] = old

    def test_get_queue_returns_rq_queue(self):
        from db.queue import get_queue
        q = get_queue()
        self.assertIsNotNone(q)
        self.assertEqual(q.name, "agent-runs")

    def test_enqueue_and_fetch_job(self):
        from db.queue import enqueue_run, fetch_job

        job_id = f"test-job-{uuid.uuid4().hex[:8]}"
        enqueue_run(_sample_job_for_test, 21, job_id=job_id)
        job = fetch_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.id, job_id)


@unittest.skipIf(REDIS_URL, "this class specifically covers the NO-REDIS fallback path")
class TestQueueUnconfigured(unittest.TestCase):

    def test_get_queue_returns_none(self):
        from db.queue import get_queue
        self.assertIsNone(get_queue())

    def test_enqueue_run_raises_clean_error(self):
        from db.queue import enqueue_run
        with self.assertRaises(RuntimeError):
            enqueue_run(lambda: None)

    def test_fetch_job_returns_none(self):
        from db.queue import fetch_job
        self.assertIsNone(fetch_job("nonexistent"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
