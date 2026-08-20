"""
Tests for testcase/eval_sets/workspace.py.

Split the same way as test_cache_queue.py/test_sandbox.py: pure-logic tests
(output parsing, dispatch) always run; the live integration tests actually
clone/checkout real sources and run real graders, and only run when
$RUN_LIVE_EVAL_TESTS=1 is set, since they need network access and can take
minutes (bugsinpy-compile builds a venv per bug; the swebench path installs
a real package). Not run by default in `pytest testcase/test_all.py`.

    RUN_LIVE_EVAL_TESTS=1 python -m pytest testcase/test_eval_workspace.py -v
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from testcase.eval_sets import workspace

RUN_LIVE = os.environ.get("RUN_LIVE_EVAL_TESTS") == "1"


class TestBugsInPyOutputParsing(unittest.TestCase):
    """
    _classify_bugsinpy_output returns (passed, setup_failed). A real
    pytest/unittest summary line means the test runner actually ran and
    reported something -- trust its pass/fail markers. No summary line at
    all (even with a traceback) means the runner never got that far, e.g.
    it crashed on its own startup -- an environment problem, not a graded
    loss (setup_failed=True), verified against a real crash on this
    machine: BugsInPy's old pytest/py pins can't import under Python 3.13.
    """

    def test_pytest_style_pass(self):
        output = "collected 3 items\n...\n3 passed in 0.12s"
        self.assertEqual(workspace._classify_bugsinpy_output(output), (True, False))

    def test_pytest_style_fail(self):
        output = "collected 3 items\nF..\n1 failed, 2 passed in 0.12s"
        self.assertEqual(workspace._classify_bugsinpy_output(output), (False, False))

    def test_unittest_style_ok(self):
        output = "test_thing (module.TestCase) ... ok\n\nRan 1 test in 0.001s\n\nOK\n"
        self.assertEqual(workspace._classify_bugsinpy_output(output), (True, False))

    def test_unittest_style_failed(self):
        output = "test_thing (module.TestCase) ... FAILED\n\nRan 1 test in 0.001s\n\nFAILED (failures=1)\n"
        self.assertEqual(workspace._classify_bugsinpy_output(output), (False, False))

    def test_traceback_with_no_summary_is_a_setup_failure_not_a_fail(self):
        # Real captured output from a live run: BugsInPy's pinned pytest/py
        # crashing on its own import under a newer Python, before running
        # a single test -- no "N passed/failed" summary ever appears.
        output = (
            "Traceback (most recent call last):\n"
            '  File ".../_pytest/_code/__init__.py", line 3, in <module>\n'
            "    from .code import Code\n"
            "AttributeError: __spec__\n"
        )
        self.assertEqual(workspace._classify_bugsinpy_output(output), (False, True))

    def test_ambiguous_output_is_not_treated_as_passed_or_setup_failed(self):
        # No summary, no traceback/error marker either -- truly no signal.
        # Must not be misreported as passed; also not confidently a setup
        # failure since nothing points at an environment problem specifically.
        self.assertEqual(workspace._classify_bugsinpy_output("nothing recognizable here"), (False, False))


class TestSwebenchSummaryDetection(unittest.TestCase):
    """
    _grade_swebench() treats "pytest produced no summary line" as
    setup_failed rather than trusting the exit code alone -- a nonzero exit
    conflates "tests ran and some failed" (a real graded loss) with "pytest
    crashed before running anything" (an environment problem). These are
    real captured outputs from a live 30-task run that were being
    misclassified as graded FAILs before this fix -- same class of bug
    _classify_bugsinpy_output() exists to avoid for BugsInPy.
    """

    def test_import_crash_during_collection_has_no_summary(self):
        # flask-4045: an incompatible transitive dep (werkzeug) installed by
        # the generic `pip install -e .`, unrelated to the agent's fix.
        output = (
            "src/flask/helpers.py:15: in <module>\n"
            "    from werkzeug.urls import url_quote\n"
            "E   ImportError: cannot import name 'url_quote' from 'werkzeug.urls'\n"
        )
        self.assertIsNone(workspace._PYTEST_SUMMARY_RE.search(output))

    def test_test_not_found_has_no_summary(self):
        # django-13448: the named test doesn't exist in this checkout.
        output = "no tests ran in 0.00s\nERROR: file or directory not found: test_migrate_test_setting_false\n"
        self.assertIsNone(workspace._PYTEST_SUMMARY_RE.search(output))

    def test_collection_error_has_no_summary(self):
        # xarray-4248
        output = "ERROR: found no collectors for /workspace/xarray/tests/test_formatting.py::TestFormatting::test_diff_attrs_repr_with_array\n"
        self.assertIsNone(workspace._PYTEST_SUMMARY_RE.search(output))

    def test_bare_error_count_is_not_a_real_summary(self):
        # xarray-4248's actual full pytest output ends with this exact line
        # ("1 warning, 1 error in 0.03s") -- a naive regex matching any
        # "N error" as "a real pytest run happened" would wrongly treat this
        # ModuleNotFoundError-during-collection as a graded result. pytest
        # reports collection-time crashes this way too, not just genuine
        # per-test failures, so "error" alone (no passed/failed) must not
        # count as evidence real tests were collected and executed.
        output = "1 warning, 1 error in 0.03s"
        self.assertIsNone(workspace._PYTEST_SUMMARY_RE.search(output))

    def test_real_pytest_run_has_a_summary(self):
        output = "collected 3 items\nF..\n1 failed, 2 passed in 0.12s"
        self.assertIsNotNone(workspace._PYTEST_SUMMARY_RE.search(output))


class TestDispatch(unittest.TestCase):
    def test_prepare_workspace_rejects_unknown_strategy(self):
        instance = {"workspace": {"strategy": "not_a_real_strategy"}}
        with self.assertRaises(ValueError):
            workspace.prepare_workspace(instance, Path("/tmp/doesnt-matter"))

    def test_grade_rejects_unknown_strategy(self):
        instance = {"workspace": {"strategy": "not_a_real_strategy"}}
        with self.assertRaises(ValueError):
            workspace.grade(instance, Path("/tmp/doesnt-matter"))


class TestGradeResult(unittest.TestCase):
    def test_setup_failed_defaults_false(self):
        result = workspace.GradeResult(passed=True)
        self.assertFalse(result.setup_failed)

    def test_truncate_leaves_short_text_alone(self):
        self.assertEqual(workspace._truncate("short"), "short")

    def test_truncate_cuts_long_text(self):
        text = "x" * 5000
        truncated = workspace._truncate(text, limit=100)
        self.assertTrue(truncated.endswith("[truncated]"))
        self.assertLess(len(truncated), len(text))


@unittest.skipUnless(RUN_LIVE, "set RUN_LIVE_EVAL_TESTS=1 to run (needs network, can take minutes)")
class TestQuixBugsLive(unittest.TestCase):
    """
    Only exercises quixbugs -- it's the fast, self-contained source (no
    venv build, no package install). bugsinpy/swebench live paths are
    exercised manually via run_portfolio_eval.py, not as a routine test,
    since a single bugsinpy-compile alone can take minutes.
    """

    @classmethod
    def setUpClass(cls):
        if not (workspace.CACHE_DIR / "quixbugs").is_dir():
            raise unittest.SkipTest("QuixBugs not cached -- run build_eval_set.py first")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="eval_ws_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _instance(self, name: str) -> dict:
        return {
            "workspace": {
                "strategy": "quixbugs_isolated",
                "target_file": f"python_programs/{name}.py",
                "test_file": f"python_testcases/test_{name}.py",
            }
        }

    def test_buggy_workspace_fails_grading(self):
        instance = self._instance("max_sublist_sum")
        dest = self.tmp / "ws"
        workspace.prepare_workspace(instance, dest)
        result = workspace.grade(instance, dest)
        self.assertFalse(result.passed)
        self.assertFalse(result.setup_failed)

    def test_prepare_succeeds_over_a_stale_leftover_workspace(self):
        """
        Regression test: a workspace left behind by an interrupted previous
        run (killed mid-run, crashed, etc.) used to make prepare_workspace()
        fail with FileExistsError on the very next attempt -- verified live,
        a run killed partway through quixbugs__depth_first_search left its
        partially-populated dest/python_programs/ in place, and the next
        run of that same instance failed at the workspace_prep stage before
        the agent ever got to run, reported as setup_failed rather than a
        graded result.
        """
        instance = self._instance("max_sublist_sum")
        dest = self.tmp / "ws"
        dest.mkdir()
        (dest / "python_programs").mkdir()
        (dest / "python_programs" / "stale_leftover.txt").write_text("from a previous run")

        workspace.prepare_workspace(instance, dest)

        self.assertFalse((dest / "python_programs" / "stale_leftover.txt").exists())
        self.assertTrue((dest / "python_programs" / "max_sublist_sum.py").exists())

    def test_workspace_does_not_expose_reference_fix(self):
        instance = self._instance("max_sublist_sum")
        dest = self.tmp / "ws"
        workspace.prepare_workspace(instance, dest)
        self.assertFalse((dest / "correct_python_programs").exists())
        self.assertFalse((dest / "correct_java_programs").exists())

    def test_fixing_the_target_file_makes_grading_pass(self):
        instance = self._instance("max_sublist_sum")
        dest = self.tmp / "ws"
        workspace.prepare_workspace(instance, dest)
        # Copy in the real fix to prove grading isn't just always-false --
        # mirrors what an agent's successful patch would produce.
        fixed = workspace.CACHE_DIR / "quixbugs" / "correct_python_programs" / "max_sublist_sum.py"
        shutil.copy2(fixed, dest / "python_programs" / "max_sublist_sum.py")
        result = workspace.grade(instance, dest)
        self.assertTrue(result.passed, result.detail)


if __name__ == "__main__":
    unittest.main()
