"""
Tests for db/store.py (Postgres persistence layer).

Split out of test_all.py on purpose: test_all.py's contract is "no external
services required" (106+ tests, pure unit/fallback-mode coverage). The live
half of this file needs a reachable Postgres with db/schema.sql applied:

    docker compose up -d postgres
    DATABASE_URL=postgresql://agent:agent_dev_password@localhost:5432/agent \
        python -m pytest testcase/test_db_store.py -v

Without $DATABASE_URL set, the live tests are skipped (not failed) and only
the "unconfigured store is a safe no-op" tests run -- so this file is still
safe to include in a default `pytest testcase/` sweep.
"""
from __future__ import annotations

import os
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATABASE_URL = os.environ.get("DATABASE_URL")


@unittest.skipUnless(DATABASE_URL, "DATABASE_URL not set -- see module docstring to run this class")
class TestDBStoreLive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from db import store
        cls.store = store
        if not store.init_pool(DATABASE_URL, timeout=5.0):
            raise unittest.SkipTest("Postgres not reachable at $DATABASE_URL")

    @classmethod
    def tearDownClass(cls):
        cls.store.close_pool()

    def setUp(self):
        self.session_id = f"test-{uuid.uuid4().hex[:8]}"
        self.run_id = f"run-{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        from db.store import _conn
        try:
            with _conn() as conn:
                conn.execute("DELETE FROM sessions WHERE session_id = %s", (self.session_id,))
                conn.execute("DELETE FROM runs WHERE run_id = %s", (self.run_id,))
        except Exception:
            pass

    def test_create_and_list_session(self):
        self.store.create_session(self.session_id, "test-repo", "/tmp/x")
        sessions = self.store.list_sessions(limit=500)
        self.assertTrue(any(s["session_id"] == self.session_id for s in sessions))

    def test_create_session_is_idempotent(self):
        # ON CONFLICT DO NOTHING -- a duplicate create must not raise.
        self.store.create_session(self.session_id, "test-repo", "/tmp/x")
        self.store.create_session(self.session_id, "test-repo", "/tmp/x")

    def test_touch_and_close_session(self):
        self.store.create_session(self.session_id, "test-repo", "/tmp/x")
        self.store.close_session(self.session_id)
        sessions = self.store.list_sessions(limit=500)
        row = next(s for s in sessions if s["session_id"] == self.session_id)
        self.assertIsNotNone(row["closed_at"])

    def test_add_and_get_messages_in_order(self):
        self.store.create_session(self.session_id, "test-repo", "/tmp/x")
        self.store.add_message(self.session_id, "user", "hello")
        self.store.add_message(self.session_id, "agent", "hi there", run_id=self.run_id)

        messages = self.store.get_messages(self.session_id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["run_id"], None)
        self.assertEqual(messages[1]["role"], "agent")
        self.assertEqual(messages[1]["run_id"], self.run_id)

    def test_create_and_finish_run(self):
        self.store.create_run(self.run_id, None, "do a thing", repo_url=None)
        run = self.store.get_run(self.run_id)
        self.assertIsNotNone(run)
        self.assertEqual(run["run_status"], "running")
        self.assertIsNone(run["finished_at"])

        self.store.finish_run(self.run_id, {
            "run_status": "completed",
            "validation": "passed",
            "stop_reason": None,
            "replan_count": 0,
            "tool_call_count": 3,
            "files_modified": ["a.py"],
            "final_answer": "done",
        })

        run = self.store.get_run(self.run_id)
        self.assertEqual(run["run_status"], "completed")
        self.assertEqual(run["tool_call_count"], 3)
        self.assertIsNotNone(run["finished_at"])

    def test_get_run_missing_returns_none(self):
        self.assertIsNone(self.store.get_run("does-not-exist"))

    def test_delete_run(self):
        self.store.create_run(self.run_id, None, "throwaway")
        self.assertTrue(self.store.delete_run(self.run_id))
        self.assertIsNone(self.store.get_run(self.run_id))
        self.assertFalse(self.store.delete_run(self.run_id))  # already gone -> False, not an error

    def test_session_delete_cascades_to_messages(self):
        self.store.create_session(self.session_id, "test-repo", "/tmp/x")
        self.store.add_message(self.session_id, "user", "hello")

        from db.store import _conn
        with _conn() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = %s", (self.session_id,))

        self.assertEqual(self.store.get_messages(self.session_id), [])

    def test_run_session_set_null_on_session_delete(self):
        self.store.create_session(self.session_id, "test-repo", "/tmp/x")
        self.store.create_run(self.run_id, self.session_id, "a task")

        from db.store import _conn
        with _conn() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = %s", (self.session_id,))

        run = self.store.get_run(self.run_id)
        self.assertIsNotNone(run)  # run itself survives (ON DELETE SET NULL, not CASCADE)
        self.assertIsNone(run["session_id"])


@unittest.skipIf(DATABASE_URL, "this class specifically covers the NO-DB fallback path")
class TestDBStoreUnconfigured(unittest.TestCase):
    """When $DATABASE_URL isn't set, every function must be a safe, silent no-op."""

    def test_not_available_without_dsn(self):
        from db import store
        self.assertFalse(store.available())

    def test_writes_do_not_raise(self):
        from db import store
        store.create_session("x", "y", "z")
        store.touch_session("x")
        store.close_session("x")
        store.add_message("x", "user", "hi")
        store.create_run("r", None, "task")
        store.finish_run("r", {"run_status": "completed"})
        store.delete_run("r")

    def test_reads_return_empty_not_none_or_error(self):
        from db import store
        self.assertEqual(store.list_sessions(), [])
        self.assertEqual(store.get_messages("x"), [])
        self.assertEqual(store.list_runs(), [])
        self.assertIsNone(store.get_run("x"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
