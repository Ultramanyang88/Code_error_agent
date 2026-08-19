"""
Complete test suite for Code Error Agent.

Run:
    python testcase/test_all.py
    python -m pytest testcase/test_all.py -v
"""
from __future__ import annotations

import sys
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.state import (
    AgentBudget, AgentState, PlanStep, ToolResult,
    StepStatus, RunStatus, ValidationStatus,
)
from core.planner import Planner
from core.executor import Executor
from core.memory import AgentMemory
from tools.tools import (
    list_files, read_file, search_code, write_file,
    replace_in_file, run_command, run_tests, identify_error, get_tool_map,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_state(task: str = "test task", repo_root: str | None = None) -> AgentState:
    """Return an AgentState pointed at a temp or given repo root."""
    root = repo_root or tempfile.mkdtemp(prefix="agent_test_")
    return AgentState(input_query=task, repo_root=root)


def make_repo(files: dict[str, str]) -> Path:
    """Create a temp directory with the given {relative_path: content} files."""
    tmp = Path(tempfile.mkdtemp(prefix="repo_"))
    for rel, content in files.items():
        target = tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return tmp


# ─────────────────────────────────────────────────────────────────────────────
# 1. State Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentBudget(unittest.TestCase):

    def test_defaults_are_positive(self):
        b = AgentBudget()
        self.assertGreater(b.max_plan_steps, 0)
        self.assertGreater(b.max_tool_calls, 0)
        self.assertGreater(b.max_replans, 0)

    def test_invalid_budget_raises(self):
        with self.assertRaises(ValueError):
            AgentBudget(max_plan_steps=0)
        with self.assertRaises(ValueError):
            AgentBudget(max_replans=-1)

    def test_frozen(self):
        b = AgentBudget()
        with self.assertRaises(Exception):
            setattr(b, "max_replans", 100)


class TestToolResult(unittest.TestCase):

    def test_to_text_success(self):
        r = ToolResult(tool_name="read_file", success=True, output="hello")
        text = r.to_text()
        self.assertIn("[SUCCESS]", text)
        self.assertIn("hello", text)

    def test_to_text_failure(self):
        r = ToolResult(tool_name="run_tests", success=False, output="", error="failed")
        text = r.to_text()
        self.assertIn("[FAILED]", text)
        self.assertIn("failed", text)

    def test_to_text_truncation(self):
        r = ToolResult(tool_name="x", success=True, output="A" * 5000)
        text = r.to_text(max_chars=100)
        self.assertLessEqual(len(text), 120)  # small buffer for header


class TestPlanStep(unittest.TestCase):

    def test_lifecycle(self):
        step = PlanStep(step_id=1, task="do something")
        self.assertEqual(step.status, StepStatus.PENDING)

        step.mark_running()
        self.assertEqual(step.status, StepStatus.RUNNING)

        step.mark_completed("done")
        self.assertTrue(step.is_completed)
        self.assertEqual(step.result, "done")

    def test_mark_failed_increments_retry(self):
        step = PlanStep(step_id=1, task="task")
        step.mark_failed("oops")
        self.assertEqual(step.retry_count, 1)
        step.mark_failed("again")
        self.assertEqual(step.retry_count, 2)


class TestAgentState(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.state = AgentState(input_query="test", repo_root=str(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_repo_path_safe(self):
        p = self.state.repo_path("foo/bar.py")
        # Use resolve() because macOS tempdir is a symlink (/var → /private/var)
        self.assertTrue(str(p).startswith(str(self.tmp.resolve())))

    def test_repo_path_traversal_blocked(self):
        with self.assertRaises(ValueError):
            self.state.repo_path("../../etc/passwd")

    def test_add_tool_result_tracks_files_read(self):
        r = ToolResult("read_file", True, "content", metadata={"path": "main.py"})
        self.state.add_tool_result(r)
        self.assertIn("main.py", self.state.files_read)

    def test_add_tool_result_tracks_files_modified(self):
        r = ToolResult("replace_in_file", True, "done",
                       metadata={"changed_files": ["core/foo.py"]})
        self.state.add_tool_result(r)
        self.assertIn("core/foo.py", self.state.files_modified)

    def test_add_tool_result_sets_validation_passed(self):
        r = ToolResult("run_tests", True, "passed")
        self.state.add_tool_result(r)
        self.assertEqual(self.state.validation_status, ValidationStatus.PASSED)

    def test_add_tool_result_sets_validation_failed(self):
        r = ToolResult("run_tests", False, "", error="1 test failed")
        self.state.add_tool_result(r)
        self.assertEqual(self.state.validation_status, ValidationStatus.FAILED)

    def test_is_finished_property(self):
        self.assertFalse(self.state.is_finished)
        self.state.run_status = RunStatus.COMPLETED
        self.assertTrue(self.state.is_finished)

    def test_get_current_step_returns_first_pending(self):
        self.state.add_plan([
            PlanStep(step_id=1, task="a"),
            PlanStep(step_id=2, task="b"),
        ])
        step = self.state.get_current_step()
        self.assertIsNotNone(step)
        assert step is not None
        self.assertEqual(step.step_id, 1)

    def test_get_current_step_skips_completed(self):
        s1 = PlanStep(step_id=1, task="a")
        s1.mark_running()
        s1.mark_completed("done")
        s2 = PlanStep(step_id=2, task="b")
        self.state.add_plan([s1, s2])
        step = self.state.get_current_step()
        self.assertIsNotNone(step)
        assert step is not None
        self.assertEqual(step.step_id, 2)

    def test_get_current_step_returns_none_when_all_done(self):
        s = PlanStep(step_id=1, task="a")
        s.mark_running()
        s.mark_completed("ok")
        self.state.add_plan([s])
        self.assertIsNone(self.state.get_current_step())

    def test_get_current_step_skips_step_with_exhausted_retries(self):
        # step 1 has failed more times than the budget allows -> permanently
        # skipped by get_current_step, but step 2 is still pending.
        s1 = PlanStep(step_id=1, task="a")
        for _ in range(self.state.budget.max_step_retries + 1):
            s1.mark_failed("boom")
        s2 = PlanStep(step_id=2, task="b")
        self.state.add_plan([s1, s2])

        step = self.state.get_current_step()
        self.assertIsNotNone(step)
        assert step is not None
        self.assertEqual(step.step_id, 2)

    def test_has_unresolved_failures_true_when_retries_exhausted(self):
        s = PlanStep(step_id=1, task="a")
        for _ in range(self.state.budget.max_step_retries + 1):
            s.mark_failed("boom")
        self.state.add_plan([s])
        self.assertIsNone(self.state.get_current_step())
        self.assertTrue(self.state.has_unresolved_failures())

    def test_has_unresolved_failures_false_when_all_completed(self):
        s = PlanStep(step_id=1, task="a")
        s.mark_running()
        s.mark_completed("ok")
        self.state.add_plan([s])
        self.assertFalse(self.state.has_unresolved_failures())

    def test_has_unresolved_failures_false_while_step_still_retryable(self):
        s = PlanStep(step_id=1, task="a")
        s.mark_failed("boom")  # one failure, still well under the retry budget
        self.state.add_plan([s])
        self.assertFalse(self.state.has_unresolved_failures())

    def test_plan_summary(self):
        self.state.add_plan([PlanStep(step_id=1, task="inspect")])
        summary = self.state.plan_summary()
        self.assertIn("inspect", summary)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tool Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestListFiles(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({
            "main.py": "print('hi')",
            "core/__init__.py": "",
            "core/state.py": "# state",
        })
        self.state = make_state(repo_root=str(self.repo))

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_lists_files(self):
        r = list_files(self.state, directory=".", max_depth=2)
        self.assertTrue(r.success)
        self.assertIn("main.py", r.output)
        self.assertIn("core", r.output)

    def test_missing_directory(self):
        r = list_files(self.state, directory="nonexistent")
        self.assertFalse(r.success)
        self.assertIn("does not exist", r.error or "")

    def test_max_depth_limits_output(self):
        # max_depth=0 shows only top-level entries (no recursion into subdirs)
        r = list_files(self.state, directory=".", max_depth=0)
        self.assertTrue(r.success)
        self.assertNotIn("state.py", r.output)
        self.assertIn("core", r.output)


class TestReadFile(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({"hello.py": "line1\nline2\nline3\n"})
        self.state = make_state(repo_root=str(self.repo))

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_reads_full_file(self):
        r = read_file(self.state, path="hello.py")
        self.assertTrue(r.success)
        self.assertIn("line1", r.output)
        self.assertIn("line3", r.output)

    def test_reads_line_range(self):
        r = read_file(self.state, path="hello.py", start_line=2, end_line=2)
        self.assertTrue(r.success)
        self.assertIn("line2", r.output)
        self.assertNotIn("line1", r.output)

    def test_missing_file(self):
        r = read_file(self.state, path="nope.py")
        self.assertFalse(r.success)
        self.assertIn("does not exist", r.error or "")

    def test_path_traversal_blocked(self):
        r = read_file(self.state, path="../../etc/passwd")
        self.assertFalse(r.success)

    def test_reads_empty_file(self):
        (self.repo / "blank.py").write_text("")
        r = read_file(self.state, path="blank.py")
        self.assertTrue(r.success)
        self.assertEqual(r.metadata["total_lines"], 0)


class TestSearchCode(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({
            "a.py": "def foo():\n    return 42\n",
            "b.py": "def bar():\n    return foo()\n",
        })
        self.state = make_state(repo_root=str(self.repo))

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_finds_keyword(self):
        r = search_code(self.state, query="def foo")
        self.assertTrue(r.success)
        self.assertIn("a.py", r.output)

    def test_no_match(self):
        r = search_code(self.state, query="zzz_nonexistent_zzz")
        self.assertTrue(r.success)
        self.assertIn("No matches", r.output)

    def test_glob_filter(self):
        r = search_code(self.state, query="def", include_glob="b.py")
        self.assertTrue(r.success)
        self.assertIn("b.py", r.output)
        self.assertNotIn("a.py", r.output)

    def test_regex_mode(self):
        r = search_code(self.state, query=r"def \w+\(\)", use_regex=True)
        self.assertTrue(r.success)
        self.assertGreater(r.metadata["num_matches"], 0)

    def test_empty_query_fails(self):
        r = search_code(self.state, query="")
        self.assertFalse(r.success)


class TestWriteAndReplaceFile(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({})
        self.state = make_state(repo_root=str(self.repo))

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_write_new_file(self):
        r = write_file(self.state, path="new.py", content="x = 1\n")
        self.assertTrue(r.success)
        self.assertTrue((self.repo / "new.py").exists())

    def test_write_no_overwrite_blocks(self):
        (self.repo / "existing.py").write_text("old")
        r = write_file(self.state, path="existing.py", content="new", overwrite=False)
        self.assertFalse(r.success)

    def test_write_overwrite(self):
        (self.repo / "existing.py").write_text("old")
        r = write_file(self.state, path="existing.py", content="new", overwrite=True)
        self.assertTrue(r.success)
        self.assertEqual((self.repo / "existing.py").read_text(), "new")

    def test_replace_in_file(self):
        (self.repo / "code.py").write_text("x = 1\ny = 2\n")
        r = replace_in_file(self.state, path="code.py", old_text="x = 1", new_text="x = 99")
        self.assertTrue(r.success)
        self.assertIn("99", (self.repo / "code.py").read_text())

    def test_replace_old_text_not_found(self):
        (self.repo / "code.py").write_text("x = 1\n")
        r = replace_in_file(self.state, path="code.py", old_text="zzz", new_text="nope")
        self.assertFalse(r.success)
        self.assertIn("not found", r.error or "")

    def test_replace_in_missing_file(self):
        r = replace_in_file(self.state, path="ghost.py", old_text="x", new_text="y")
        self.assertFalse(r.success)


class TestRunTestsDiscovery(unittest.TestCase):
    """
    Regression coverage for run_tests() finding tests that aren't under a
    tests/ or testcase/ folder -- e.g. a flat test_*.py sitting next to the
    module it tests, which is exactly how the eval fixtures under
    testcase/py/ and testcase/tasks/*/ are laid out. Before this fix,
    run_tests() silently degraded to `compileall` (a syntax check only) for
    any repo shaped like that, so a real failing test could still report
    validation_status=PASSED.
    """

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_finds_flat_test_file_with_no_tests_folder(self):
        self.repo = make_repo({
            "calc.py": "def add(a, b):\n    return a + b\n",
            "test_calc.py": (
                "from calc import add\n"
                "def test_add():\n"
                "    assert add(1, 2) == 3\n"
            ),
        })
        state = make_state(repo_root=str(self.repo))
        r = run_tests(state)
        self.assertTrue(r.success, r.output)
        self.assertIn("1 passed", r.output)

    def test_reports_failure_not_false_positive_pass(self):
        # The exact failure mode this bug caused: a genuinely broken flat
        # test file must come back as a FAILED run_tests, not silently pass
        # via a compileall fallback that never executed the assertion.
        self.repo = make_repo({
            "calc.py": "def add(a, b):\n    return a - b\n",  # bug: subtracts
            "test_calc.py": (
                "from calc import add\n"
                "def test_add():\n"
                "    assert add(1, 2) == 3\n"
            ),
        })
        state = make_state(repo_root=str(self.repo))
        r = run_tests(state)
        self.assertFalse(r.success)
        self.assertIn("1 failed", r.output)

    def test_falls_back_to_compileall_when_genuinely_no_tests(self):
        self.repo = make_repo({"script.py": "x = 1\n"})
        state = make_state(repo_root=str(self.repo))
        r = run_tests(state)
        self.assertTrue(r.success, r.output)
        self.assertIn("Compiling", r.output)


class TestRunCommand(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({"hello.py": "print('hi')"})
        self.state = make_state(repo_root=str(self.repo))

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_safe_command(self):
        r = run_command(self.state, command="echo hello")
        self.assertTrue(r.success)
        self.assertIn("hello", r.output)

    def test_dangerous_command_blocked(self):
        r = run_command(self.state, command="sudo rm -rf /")
        self.assertFalse(r.success)
        self.assertIn("Blocked", r.error or "")

    def test_exit_code_nonzero(self):
        r = run_command(self.state, command="python -c 'exit(1)'")
        self.assertFalse(r.success)

    def test_python_script(self):
        r = run_command(self.state, command="python hello.py")
        self.assertTrue(r.success)
        self.assertIn("hi", r.output)


class TestSandboxConfig(unittest.TestCase):
    """
    Fast, Docker-free coverage: env var parsing and the disabled-by-default
    behavior. Live container tests (start/exec/network isolation/cleanup)
    are in testcase/test_sandbox.py, gated on Docker actually being
    reachable -- see that file's docstring.
    """

    def setUp(self):
        self._orig = os.environ.get("AGENT_SANDBOX_ENABLED")

    def tearDown(self):
        if self._orig is None:
            os.environ.pop("AGENT_SANDBOX_ENABLED", None)
        else:
            os.environ["AGENT_SANDBOX_ENABLED"] = self._orig

    def test_disabled_by_default(self):
        os.environ.pop("AGENT_SANDBOX_ENABLED", None)
        from sandbox.docker_sandbox import is_sandbox_enabled
        self.assertFalse(is_sandbox_enabled())

    def test_enabled_by_truthy_values(self):
        from sandbox.docker_sandbox import is_sandbox_enabled
        for value in ["1", "true", "True", "yes", "YES"]:
            os.environ["AGENT_SANDBOX_ENABLED"] = value
            self.assertTrue(is_sandbox_enabled(), f"expected enabled for {value!r}")

    def test_disabled_by_falsy_values(self):
        from sandbox.docker_sandbox import is_sandbox_enabled
        for value in ["0", "false", "", "no"]:
            os.environ["AGENT_SANDBOX_ENABLED"] = value
            self.assertFalse(is_sandbox_enabled(), f"expected disabled for {value!r}")

    def test_run_command_uses_host_path_when_sandbox_disabled(self):
        # Regression guard for the sys.executable-inside-a-container bug:
        # with sandboxing off (the default), _python_command() must return
        # this exact interpreter, not a bare "python3".
        os.environ.pop("AGENT_SANDBOX_ENABLED", None)
        from tools.tools import _python_command
        import sys as _sys
        self.assertEqual(_python_command(), _sys.executable)

    def test_container_name_is_stable_across_processes(self):
        # Regression guard: container_name must NOT be derived from Python's
        # built-in hash() (salted per-process by PYTHONHASHSEED) -- otherwise
        # a container orphaned by a crash (skipping the atexit cleanup) can
        # never be found/removed by a later process, since the name it would
        # look for keeps changing. Simulate "another process" by checking the
        # name is deterministic from repo_root alone, independent of hash().
        from sandbox.docker_sandbox import DockerSandbox
        a = DockerSandbox(repo_root="/tmp/some/repo")
        b = DockerSandbox(repo_root="/tmp/some/repo")
        self.assertEqual(a.container_name, b.container_name)
        # and different repos must not collide
        c = DockerSandbox(repo_root="/tmp/some/other-repo")
        self.assertNotEqual(a.container_name, c.container_name)


class TestIdentifyError(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({})
        self.state = make_state(repo_root=str(self.repo))

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_module_not_found(self):
        r = identify_error(self.state, error_log="ModuleNotFoundError: No module named 'foo'")
        self.assertTrue(r.success)
        self.assertEqual(r.metadata["error_type"], "ModuleNotFoundError")

    def test_syntax_error(self):
        r = identify_error(self.state, error_log='SyntaxError: invalid syntax\n  File "a.py", line 3')
        self.assertTrue(r.success)
        self.assertEqual(r.metadata["error_type"], "SyntaxError")
        self.assertTrue(len(r.metadata["relevant_files"]) > 0)

    def test_empty_log_fails(self):
        r = identify_error(self.state, error_log="")
        self.assertFalse(r.success)

    def test_attribute_error(self):
        r = identify_error(self.state, error_log="AttributeError: 'NoneType' has no attr 'x'")
        self.assertTrue(r.success)
        self.assertEqual(r.metadata["error_type"], "AttributeError")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Executor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutorValidation(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({"main.py": "x = 1\n"})
        self.state = make_state(repo_root=str(self.repo))
        self.executor = Executor(client=None, tools=get_tool_map())

    def tearDown(self):
        shutil.rmtree(Path(self.state.repo_root), ignore_errors=True)

    def test_valid_args_returns_none(self):
        err = self.executor._validate_tool_arg("read_file", {"path": "main.py"})
        self.assertIsNone(err)

    def test_missing_required_returns_error(self):
        err = self.executor._validate_tool_arg("read_file", {})
        self.assertIsNotNone(err)
        self.assertIn("path", err)

    def test_none_value_is_missing(self):
        err = self.executor._validate_tool_arg("read_file", {"path": None})
        self.assertIsNotNone(err)

    def test_false_value_is_not_missing(self):
        # "use_regex=False" must NOT be flagged as missing
        err = self.executor._validate_tool_arg("search_code", {"query": "x", "use_regex": False})
        self.assertIsNone(err)

    def test_zero_value_is_not_missing(self):
        err = self.executor._validate_tool_arg("search_code", {"query": "x", "max_results": 0})
        self.assertIsNone(err)


class TestExecutorNormalize(unittest.TestCase):

    def setUp(self):
        self.executor = Executor(client=None)

    def test_list_files_path_alias(self):
        args = self.executor._normalize_tool_arguments("list_files", {"path": "src"})
        self.assertEqual(args["directory"], "src")
        self.assertNotIn("path", args)

    def test_read_file_file_path_alias(self):
        args = self.executor._normalize_tool_arguments("read_file", {"file_path": "foo.py"})
        self.assertEqual(args["path"], "foo.py")

    def test_search_code_defaults(self):
        args = self.executor._normalize_tool_arguments("search_code", {"query": "fn"})
        self.assertEqual(args["include_glob"], "*.py")
        self.assertFalse(args["use_regex"])

    def test_retrieve_context_question_alias(self):
        args = self.executor._normalize_tool_arguments("retrieve_context", {"question": "what is X"})
        self.assertEqual(args["query"], "what is X")

    def test_run_command_cmd_alias(self):
        args = self.executor._normalize_tool_arguments("run_command", {"cmd": "ls"})
        self.assertEqual(args["command"], "ls")


class TestToolCategories(unittest.TestCase):

    def test_every_tool_has_a_known_category(self):
        from tools.specs import TOOL_SPECS, TOOL_CATEGORIES
        for name, spec in TOOL_SPECS.items():
            self.assertIn(spec.get("category"), TOOL_CATEGORIES, f"{name} has no valid category")

    def test_category_of_known_and_unknown_tool(self):
        from tools.specs import category_of
        self.assertEqual(category_of("read_file"), "inspection")
        self.assertIsNone(category_of("mcp_fs__read_file"))  # not in TOOL_SPECS

    def test_expand_by_category_pulls_in_siblings(self):
        from tools.specs import expand_by_category
        # write_file's category (mutation) also contains replace_in_file/apply_patch
        expanded = expand_by_category(["write_file"])
        self.assertIn("write_file", expanded)
        self.assertIn("replace_in_file", expanded)
        self.assertIn("apply_patch", expanded)
        self.assertNotIn("run_command", expanded)  # different category

    def test_expand_by_category_passthrough_for_unrecognized_tools(self):
        from tools.specs import expand_by_category
        expanded = expand_by_category(["mcp_fs__read_file"], available=["mcp_fs__read_file"])
        self.assertEqual(expanded, ["mcp_fs__read_file"])

    def test_tool_descriptions_narrows_to_suggested_category(self):
        executor = Executor(client=None, tools=get_tool_map())
        narrowed = executor._tool_descriptions(suggested_tools=["write_file"])
        self.assertIn("write_file", narrowed)
        self.assertIn("apply_patch", narrowed)      # same category, pulled in
        self.assertNotIn("run_command", narrowed)   # different category, excluded

    def test_tool_descriptions_full_list_without_suggestion(self):
        executor = Executor(client=None, tools=get_tool_map())
        full = executor._tool_descriptions()
        self.assertIn("write_file", full)
        self.assertIn("run_command", full)


class TestToolRegistry(unittest.TestCase):
    """
    ToolRegistry merges tools/specs.py's static built-in categories with
    per-server categories declared on MCPServerConfig, so MCP tools aren't
    invisible to the same routing system built-in tools use.
    """

    def _make_registry(self):
        from tools.registry import ToolRegistry
        from agent_mcp.config import MCPServerConfig

        all_tools = dict(get_tool_map())
        all_tools.update({
            "gh__search_issues": lambda **kw: None,
            "gh__create_pull_request": lambda **kw: None,
            "misc__ping": lambda **kw: None,  # server with no declared category at all
        })
        gh_config = MCPServerConfig(
            name="gh", command="npx", namespace="gh",
            category="vcs", category_purpose="Git hosting operations: issues, PRs, reviews.",
            tool_categories={"create_pull_request": "mutation"},  # per-tool override
        )
        misc_config = MCPServerConfig(name="misc", command="npx", namespace="misc")  # no category
        return ToolRegistry.build(all_tools, mcp_configs=[gh_config, misc_config])

    def test_static_tool_category_still_resolves(self):
        registry = self._make_registry()
        self.assertEqual(registry.category_of("read_file"), "inspection")

    def test_mcp_tool_gets_server_default_category(self):
        registry = self._make_registry()
        self.assertEqual(registry.category_of("gh__search_issues"), "vcs")

    def test_mcp_tool_per_tool_override_wins_over_server_default(self):
        registry = self._make_registry()
        self.assertEqual(registry.category_of("gh__create_pull_request"), "mutation")

    def test_unmapped_mcp_tool_has_no_category(self):
        registry = self._make_registry()
        self.assertIsNone(registry.category_of("misc__ping"))

    def test_category_summary_prompt_includes_new_mcp_category(self):
        registry = self._make_registry()
        summary = registry.category_summary_prompt()
        self.assertIn("vcs", summary)
        self.assertIn("Git hosting operations", summary)
        self.assertIn("uncategorized", summary)  # misc__ping falls here

    def test_expand_by_category_accepts_bare_category_name(self):
        registry = self._make_registry()
        # planner names a category directly instead of a specific tool
        expanded = registry.expand_by_category(["mutation"])
        self.assertIn("write_file", expanded)
        self.assertIn("apply_patch", expanded)
        self.assertIn("gh__create_pull_request", expanded)  # MCP tool, same category
        self.assertNotIn("run_command", expanded)

    def test_expand_by_category_mixes_tool_and_category_names(self):
        registry = self._make_registry()
        expanded = registry.expand_by_category(["read_file", "vcs"])
        self.assertIn("read_file", expanded)
        self.assertIn("gh__search_issues", expanded)

    def test_expand_by_category_unrecognized_passthrough(self):
        registry = self._make_registry()
        expanded = registry.expand_by_category(["misc__ping"])
        self.assertEqual(expanded, ["misc__ping"])


class TestPlannerToolRegistryIntegration(unittest.TestCase):

    def test_no_registry_uses_hardcoded_tool_list(self):
        planner = Planner(client=None)
        prompt = planner._system_prompt()
        self.assertIn("Use only available tool names", prompt)

    def test_registry_uses_category_summary_instead_of_flat_list(self):
        from tools.registry import ToolRegistry
        registry = ToolRegistry.build(get_tool_map())
        planner = Planner(client=None, tool_registry=registry)
        prompt = planner._system_prompt()
        self.assertIn("CATEGORIES", prompt)
        self.assertNotIn("Use only available tool names", prompt)

        block = planner._available_tools_prompt_block()
        self.assertIn("mutation", block)
        self.assertIn("execution", block)

    def test_general_purpose_framing_present(self):
        # Planner should no longer read as bug-fix-only.
        planner = Planner(client=None)
        prompt = planner._system_prompt()
        self.assertIn("general-purpose", prompt)
        self.assertIn("refactoring", prompt)


class TestToolResultIsEnough(unittest.TestCase):
    """
    Regression coverage for _tool_result_is_enough's used_tools tracking.

    Before the fix, _llm_execute_step never passed used_tools, so this always
    saw an empty set and summary-type steps could never be marked "enough"
    after actually reading/retrieving context.
    """

    def setUp(self):
        self.executor = Executor(client=None)
        self.summary_step = PlanStep(step_id=1, task="Summarize the project")

    def test_summary_step_not_enough_without_prior_context_tool(self):
        result = ToolResult(tool_name="read_file", success=True, output="content")
        enough = self.executor._tool_result_is_enough(
            self.summary_step, result, used_tools=set()
        )
        self.assertFalse(enough)

    def test_summary_step_enough_after_context_tool_used(self):
        # A prior round already called retrieve_context (tracked via
        # used_tools); a later, unrelated successful tool call can now be
        # treated as "this step has enough grounding to stop".
        result = ToolResult(tool_name="identify_error", success=True, output="ok")
        enough = self.executor._tool_result_is_enough(
            self.summary_step, result, used_tools={"retrieve_context"}
        )
        self.assertTrue(enough)

    def test_failed_result_is_never_enough(self):
        result = ToolResult(tool_name="read_file", success=False, output="", error="nope")
        enough = self.executor._tool_result_is_enough(
            self.summary_step, result, used_tools={"retrieve_context"}
        )
        self.assertFalse(enough)


class TestExecutorFallback(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({
            "main.py": "print('hello')\n",
            "utils.py": "def add(a, b): return a + b\n",
        })
        self.state = make_state(task="analyze the repo", repo_root=str(self.repo))
        self.executor = Executor(client=None, tools=get_tool_map())

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_fallback_list_step(self):
        step = PlanStep(step_id=1, task="Inspect repository structure",
                        suggested_tools=["list_files"])
        self.state.add_plan([step])
        result_state = self.executor.execute_current_step(self.state)
        step = result_state.plan[0]
        self.assertEqual(step.status, StepStatus.COMPLETED)

    def test_fallback_read_step(self):
        step = PlanStep(step_id=1, task="Read main.py to inspect it",
                        suggested_tools=["read_file"])
        self.state.add_plan([step])
        result_state = self.executor.execute_current_step(self.state)
        self.assertEqual(result_state.plan[0].status, StepStatus.COMPLETED)

    def test_fallback_search_step(self):
        step = PlanStep(step_id=1, task="Search for function definitions",
                        suggested_tools=["search_code"])
        self.state.add_plan([step])
        result_state = self.executor.execute_current_step(self.state)
        self.assertEqual(result_state.plan[0].status, StepStatus.COMPLETED)

    def test_execute_unknown_tool_returns_failure(self):
        result = self.executor._execute_tool("nonexistent_tool", {}, self.state)
        self.assertFalse(result.success)
        self.assertIn("not found", result.error or "")

    def test_execute_replace_in_file(self):
        result = self.executor._execute_tool(
            "replace_in_file",
            {"path": "main.py", "old_text": "print('hello')", "new_text": "print('world')"},
            self.state,
        )
        self.assertTrue(result.success)
        self.assertIn("world", (self.repo / "main.py").read_text())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Planner Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlannerFallback(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({"main.py": "x = 1"})
        self.planner = Planner(client=None)

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_creates_plan_for_analysis_task(self):
        # Avoid "implement" substring which triggers is_code_change_task
        state = make_state(task="Analyze this repo and describe the project structure",
                           repo_root=str(self.repo))
        steps = self.planner.create_initial_plan(state)
        self.assertGreater(len(steps), 0)
        # Analysis plan should NOT contain apply_patch
        tool_lists = [s.suggested_tools for s in steps]
        all_tools = [t for ts in tool_lists for t in ts]
        self.assertNotIn("apply_patch", all_tools)

    def test_creates_plan_for_fix_task(self):
        state = make_state(task="Fix the bug in eval_module.py",
                           repo_root=str(self.repo))
        steps = self.planner.create_initial_plan(state)
        all_tools = [t for s in steps for t in s.suggested_tools]
        self.assertTrue(
            any(t in all_tools for t in ("apply_patch", "replace_in_file", "run_tests")),
            msg=f"Expected editing/test tools; got {all_tools}",
        )

    def test_step_ids_are_sequential(self):
        state = make_state(task="Fix something", repo_root=str(self.repo))
        steps = self.planner.create_initial_plan(state)
        for i, s in enumerate(steps, start=1):
            self.assertEqual(s.step_id, i)

    def test_adjust_plan_increments_replan_count(self):
        state = make_state(task="Fix the bug", repo_root=str(self.repo))
        self.planner.create_initial_plan(state)
        original_count = state.replan_count
        self.planner.adjust_plan(state)
        self.assertEqual(state.replan_count, original_count + 1)

    def test_adjust_plan_extends_plan(self):
        state = make_state(task="Fix something", repo_root=str(self.repo))
        self.planner.create_initial_plan(state)
        original_len = len(state.plan)
        self.planner.adjust_plan(state)
        self.assertGreater(len(state.plan), original_len)

    # ── task taxonomy: general-purpose planning beyond bug-fix ──────────────

    def test_classify_test_writing(self):
        self.assertEqual(self.planner._classify_task("write unit tests for the parser"), "test_writing")

    def test_classify_dependency_upgrade(self):
        self.assertEqual(self.planner._classify_task("upgrade requests to the latest version"), "dependency_upgrade")

    def test_classify_refactor(self):
        self.assertEqual(self.planner._classify_task("refactor the executor module"), "refactor")

    def test_classify_feature_add(self):
        self.assertEqual(self.planner._classify_task("implement a new caching feature"), "feature_add")

    def test_classify_review(self):
        self.assertEqual(self.planner._classify_task("please review this diff"), "review")

    def test_classify_defaults_to_bug_fix(self):
        self.assertEqual(self.planner._classify_task("the server keeps crashing"), "bug_fix")

    def test_test_writing_plan_includes_write_and_run_tests(self):
        state = make_state(task="Write unit tests for the parser module", repo_root=str(self.repo))
        steps = self.planner.create_initial_plan(state)
        all_tools = [t for s in steps for t in s.suggested_tools]
        self.assertIn("run_tests", all_tools)
        self.assertTrue(any(t in all_tools for t in ("write_file", "replace_in_file")))

    def test_refactor_plan_validates_before_and_after(self):
        state = make_state(task="Refactor the memory module to reduce duplication", repo_root=str(self.repo))
        steps = self.planner.create_initial_plan(state)
        run_tests_count = sum(1 for s in steps for t in s.suggested_tools if t == "run_tests")
        # A refactor plan should validate BEFORE (baseline) and AFTER the change.
        self.assertGreaterEqual(run_tests_count, 2)

    def test_review_plan_never_edits_code(self):
        state = make_state(task="Please review this diff for correctness", repo_root=str(self.repo))
        steps = self.planner.create_initial_plan(state)
        all_tools = [t for s in steps for t in s.suggested_tools]
        for edit_tool in ("apply_patch", "write_file", "replace_in_file"):
            self.assertNotIn(edit_tool, all_tools)
        self.assertIn("git_diff", all_tools)

    def test_dependency_upgrade_plan_checks_usages_before_bumping(self):
        state = make_state(task="Upgrade the requests dependency to the latest version", repo_root=str(self.repo))
        steps = self.planner.create_initial_plan(state)
        self.assertIn("run_tests", [t for s in steps for t in s.suggested_tools])

    def test_parse_valid_json_plan(self):
        plan_json = json.dumps([
            {"task": "Read file", "reason": "need to inspect",
             "expected_output": "file contents", "suggested_tools": ["read_file"]},
            {"task": "Fix bug", "reason": "bug found",
             "expected_output": "patched code", "suggested_tools": ["replace_in_file"]},
        ])
        steps = self.planner._parse_plan(plan_json)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].task, "Read file")

    def test_parse_invalid_json_returns_empty(self):
        steps = self.planner._parse_plan("not json at all")
        self.assertEqual(steps, [])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Memory Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentMemory(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.memory = AgentMemory(short_term_limit=5, persist_dir=str(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_tool_result_short_term(self):
        r = ToolResult("list_files", True, "some output")
        self.memory.add_tool_result(r)
        self.assertEqual(len(self.memory.short_term), 1)

    def test_short_term_limit_enforced(self):
        for i in range(10):
            self.memory.add_tool_result(ToolResult(f"tool_{i}", True, f"out_{i}"))
        self.assertLessEqual(len(self.memory.short_term), 5)

    def test_add_insight_persists(self):
        self.memory.add_insight("main.py is the entry point", source="read_file")
        persist_file = self.tmp / "memory.jsonl"
        self.assertTrue(persist_file.exists())
        lines = persist_file.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertIn("entry point", data["content"])

    def test_retrieve_relevant_keyword_match(self):
        self.memory.add_insight("executor handles tool calls and steps")
        self.memory.add_insight("planner decomposes user requests")
        results = self.memory.retrieve_relevant("executor tool", top_k=3)
        self.assertEqual(results[0].content, "executor handles tool calls and steps")

    def test_retrieve_relevant_no_match(self):
        self.memory.add_insight("something unrelated")
        results = self.memory.retrieve_relevant("zzz_no_match_zzz")
        self.assertEqual(results, [])

    def test_summarize_short_term_empty(self):
        text = self.memory.summarize_short_term()
        self.assertIn("No recent", text)

    def test_summarize_short_term_content(self):
        self.memory.add_tool_result(ToolResult("read_file", True, "hello content"))
        text = self.memory.summarize_short_term()
        self.assertIn("read_file", text)

    def test_reload_from_disk(self):
        self.memory.add_insight("cached insight", memory_type="repo_insight")
        # Fresh memory object loading from the same dir
        memory2 = AgentMemory(persist_dir=str(self.tmp))
        self.assertEqual(len(memory2.long_term), 1)
        self.assertEqual(memory2.long_term[0].content, "cached insight")

    def test_update_from_tool_result_generates_insight(self):
        r = ToolResult("run_tests", True, "all passed")
        self.memory.update_from_tool_result(r)
        # Should auto-add insight about tests passing
        self.assertTrue(any("passed" in i.content for i in self.memory.long_term))

    def test_remember_preference_creates_new(self):
        item = self.memory.remember_preference("test_command", "pytest -x")
        self.assertEqual(item.memory_type, "preference")
        self.assertEqual(item.metadata["key"], "test_command")
        self.assertEqual(item.metadata["update_count"], 1)

    def test_remember_preference_upserts_same_key(self):
        first = self.memory.remember_preference("test_command", "pytest -x")
        second = self.memory.remember_preference("test_command", "pytest -x -q")

        prefs = [i for i in self.memory.long_term if i.metadata.get("key") == "test_command"]
        self.assertEqual(len(prefs), 1)  # updated in place, not duplicated
        self.assertEqual(prefs[0].content, "pytest -x -q")
        self.assertEqual(prefs[0].metadata["update_count"], 2)
        self.assertIs(first, second)  # same object, mutated

    def test_remember_preference_reload_resolves_to_latest(self):
        self.memory.remember_preference("editor", "vim")
        self.memory.remember_preference("editor", "neovim")

        # The JSONL log now has two lines for the same key; a fresh load
        # must resolve down to just the latest value, not both.
        reloaded = AgentMemory(persist_dir=str(self.tmp))
        prefs = [i for i in reloaded.long_term if i.metadata.get("key") == "editor"]
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0].content, "neovim")

    def test_retrieve_relevant_skips_inactive_status(self):
        self.memory.add_insight("stale note about the old auth flow")
        self.memory.long_term[0].status = "stale"
        results = self.memory.retrieve_relevant("old auth flow")
        self.assertEqual(results, [])

    def test_retrieve_relevant_filters_by_memory_type(self):
        self.memory.add_insight("a repo fact", memory_type="repo_insight")
        self.memory.remember_preference("style", "tabs not spaces")
        results = self.memory.retrieve_relevant("tabs style", memory_types=["preference"])
        self.assertTrue(all(r.memory_type == "preference" for r in results))


# ─────────────────────────────────────────────────────────────────────────────
# 6. End-to-End Agent Smoke Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentEndToEnd(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({
            "hello.py": "def greet(name):\n    return f'Hello, {name}!'\n",
            "test_hello.py": (
                "from hello import greet\n"
                "def test_greet():\n"
                "    assert greet('world') == 'Hello, world!'\n"
            ),
        })

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_analysis_task_completes(self):
        from main import run_agent
        state = run_agent(
            task_description="Analyze this repository and list all Python functions.",
            repo_root=str(self.repo),
            client=None,
        )
        # Fallback mode: should complete or reach replan limit
        self.assertIsNotNone(state)
        self.assertIsNotNone(state.plan)
        self.assertGreater(len(state.plan), 0)

    def test_callback_events_fired(self):
        from main import run_agent
        events = []

        def callback(event_type, _data):
            events.append(event_type)

        run_agent(
            task_description="List all Python files.",
            repo_root=str(self.repo),
            client=None,
            step_callback=callback,
        )
        self.assertIn("plan_created", events)
        self.assertIn("step_start", events)
        self.assertIn("done", events)

    def test_state_run_id_matches_caller_run_id(self):
        # state.run_id defaults to its own uuid4 -- run_agent must overwrite it
        # with the run_id it was given/returns, otherwise log events keyed on
        # state.run_id can never be correlated back to a specific API run.
        from main import run_agent
        state = run_agent(
            task_description="List all Python files.",
            repo_root=str(self.repo),
            client=None,
            run_id="fixed-test-run-id",
        )
        self.assertEqual(state.run_id, "fixed-test-run-id")

    def test_fix_bug_task(self):
        """Agent can apply a fix to a file with a known bug."""
        from main import run_agent
        # Create a file with a clear bug
        buggy = self.repo / "calc.py"
        buggy.write_text("def divide(a, b):\n    return a / b\n")

        state = run_agent(
            task_description=(
                "Fix the divide() function in calc.py: "
                "add a guard that raises ValueError when b == 0."
            ),
            repo_root=str(self.repo),
            client=None,
        )
        self.assertIsNotNone(state)
        # In fallback mode the agent won't write real code,
        # but should not crash and should produce a plan
        self.assertGreater(len(state.plan), 0)

    def test_trace_path_written(self):
        from main import run_agent
        trace_file = self.repo / "trace.jsonl"
        run_agent(
            task_description="List Python files.",
            repo_root=str(self.repo),
            client=None,
            trace_path=str(trace_file),
        )
        if trace_file.exists():
            lines = trace_file.read_text().splitlines()
            self.assertGreater(len(lines), 0)
            entry = json.loads(lines[0])
            self.assertIn("step_id", entry)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Eval Harness Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalSetup(unittest.TestCase):

    def test_tasks_directory_exists(self):
        tasks_dir = PROJECT_ROOT / "testcase" / "tasks"
        self.assertTrue(tasks_dir.exists(), f"tasks/ dir missing: {tasks_dir}")

    def test_each_task_has_required_files(self):
        tasks_dir = PROJECT_ROOT / "testcase" / "tasks"
        for task_dir in tasks_dir.iterdir():
            if not task_dir.is_dir():
                continue
            self.assertTrue((task_dir / "description.txt").exists(),
                            f"{task_dir.name}: missing description.txt")
            self.assertTrue((task_dir / "expected_outcome.json").exists(),
                            f"{task_dir.name}: missing expected_outcome.json")

    def test_expected_outcome_json_valid(self):
        tasks_dir = PROJECT_ROOT / "testcase" / "tasks"
        for task_dir in tasks_dir.iterdir():
            meta_path = task_dir / "expected_outcome.json"
            if not meta_path.exists():
                continue
            with self.subTest(task=task_dir.name):
                meta = json.loads(meta_path.read_text())
                self.assertIn("task_id", meta)
                self.assertIn("expected_validation", meta)

    def test_buggy_files_are_files_not_dirs(self):
        tasks_dir = PROJECT_ROOT / "testcase" / "tasks"
        for task_dir in tasks_dir.iterdir():
            buggy_dir = task_dir / "buggy_files"
            if not buggy_dir.exists():
                continue
            with self.subTest(task=task_dir.name):
                for f in buggy_dir.iterdir():
                    if not f.name.startswith("__"):
                        self.assertTrue(f.is_file() or f.is_dir(),
                                        f"Unexpected: {f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. API Server Unit Tests (no server needed — test request handling)
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIHelpers(unittest.TestCase):

    def test_write_uploaded_py_file(self):
        from api.server import _write_uploaded_file
        tmp = Path(tempfile.mkdtemp())
        try:
            _write_uploaded_file(b"x = 1\n", "test.py", tmp)
            self.assertTrue((tmp / "test.py").exists())
            self.assertEqual((tmp / "test.py").read_text(), "x = 1\n")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_write_uploaded_zip_file(self):
        import io, zipfile
        from api.server import _write_uploaded_file
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("inner.py", "y = 2\n")
        tmp = Path(tempfile.mkdtemp())
        try:
            _write_uploaded_file(buf.getvalue(), "upload.zip", tmp)
            self.assertTrue((tmp / "inner.py").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_clone_repo_invalid_url(self):
        from api.server import _clone_repo
        tmp = Path(tempfile.mkdtemp())
        try:
            with self.assertRaises((ValueError, RuntimeError)):
                _clone_repo("not-a-url", tmp / "repo")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Skill Registry Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillRegistry(unittest.TestCase):

    def setUp(self):
        self.skills_dir = Path(tempfile.mkdtemp(prefix="skills_"))
        # Write two test skill files
        (self.skills_dir / "fix_import.md").write_text(
            "---\n"
            "name: fix_import\n"
            "trigger_keywords: [ModuleNotFoundError, ImportError, cannot import]\n"
            "summary: Fix a broken Python import by tracing the missing module.\n"
            "---\n\n"
            "## Procedure\n"
            "1. Run identify_error.\n"
            "2. Use search_code to find the actual file.\n"
            "3. Fix the import with replace_in_file.\n"
        )
        (self.skills_dir / "debug_test.md").write_text(
            "---\n"
            "name: debug_test\n"
            "trigger_keywords: [FAILED, AssertionError, pytest]\n"
            "summary: Debug a failing pytest test by reading the test and source.\n"
            "---\n\n"
            "## Procedure\n"
            "1. Run identify_error on the pytest output.\n"
            "2. Read the failing test file.\n"
            "3. Trace back to the source function and apply a minimal fix.\n"
        )

    def tearDown(self):
        shutil.rmtree(self.skills_dir, ignore_errors=True)

    def setUp_registry(self):
        from skills.registry import SkillRegistry
        return SkillRegistry(skills_dir=str(self.skills_dir))

    def test_loads_skills(self):
        reg = self.setUp_registry()
        self.assertEqual(len(reg.skills), 2)

    def test_skill_attributes(self):
        reg = self.setUp_registry()
        names = {s.name for s in reg.skills}
        self.assertIn("fix_import", names)
        self.assertIn("debug_test", names)

    def test_find_relevant_import_error(self):
        reg = self.setUp_registry()
        results = reg.find_relevant(
            query="ModuleNotFoundError no module named foo",
            errors=["ModuleNotFoundError: No module named 'foo'"],
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].name, "fix_import")

    def test_find_relevant_test_failure(self):
        reg = self.setUp_registry()
        results = reg.find_relevant(
            query="pytest FAILED assertion",
            errors=["AssertionError: expected 1 got 2"],
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].name, "debug_test")

    def test_find_relevant_no_match(self):
        reg = self.setUp_registry()
        results = reg.find_relevant(query="completely unrelated topic xyz", errors=[])
        self.assertEqual(results, [])

    def test_skill_full_text_contains_procedure(self):
        reg = self.setUp_registry()
        skill = next(s for s in reg.skills if s.name == "fix_import")
        self.assertIn("identify_error", skill.full_text)

    def test_missing_dir_loads_empty(self):
        from skills.registry import SkillRegistry
        reg = SkillRegistry(skills_dir="/nonexistent/path/to/skills")
        self.assertEqual(len(reg.skills), 0)

    def test_malformed_md_skipped(self):
        # File without --- frontmatter should be silently ignored
        (self.skills_dir / "bad.md").write_text("no frontmatter here\n")
        reg = self.setUp_registry()
        self.assertEqual(len(reg.skills), 2)  # bad.md not loaded

    def test_real_skills_directory_loads(self):
        from skills.registry import SkillRegistry
        real_dir = PROJECT_ROOT / "skills"
        if real_dir.exists():
            reg = SkillRegistry(skills_dir=str(real_dir))
            self.assertGreater(len(reg.skills), 0,
                               msg="skills/ directory exists but loaded 0 skills")


# ─────────────────────────────────────────────────────────────────────────────
# 10. MCP Client Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPClient(unittest.TestCase):

    def test_import_succeeds(self):
        from agent_mcp.client import MCPToolClient
        self.assertIsNotNone(MCPToolClient)

    def test_client_initializes(self):
        from agent_mcp.client import MCPToolClient
        client = MCPToolClient(
            command="echo",
            args=["hello"],
            namespace="test_ns",
        )
        self.assertEqual(client.namespace, "test_ns")
        self.assertEqual(client.command, "echo")
        self.assertEqual(client.args, ["hello"])

    def test_invalid_command_raises_on_get_tool_map(self):
        from agent_mcp.client import MCPToolClient
        client = MCPToolClient(
            command="nonexistent_binary_xyz",
            args=[],
            namespace="ns",
        )
        with self.assertRaises(Exception):
            client.get_tool_map()

    def test_mcp_server_importable(self):
        from agent_mcp.server import app
        self.assertIsNotNone(app)

    def test_namespace_prefixing(self):
        # Verify the namespace convention: "ns__toolname"
        from agent_mcp.client import MCPToolClient
        client = MCPToolClient(command="echo", args=[], namespace="myns")
        self.assertEqual(client.namespace, "myns")

    def test_not_connected_until_connect_called(self):
        from agent_mcp.client import MCPToolClient
        client = MCPToolClient(command="echo", args=[], namespace="ns")
        self.assertFalse(client.is_connected)

    def test_shared_client_does_not_cache_failed_connection(self):
        # A failed connect() must not poison the process-wide cache -- the
        # next call for the same key should retry from scratch rather than
        # permanently remembering "this server is down".
        import agent_mcp.client as mcp_client_module

        key = ("nonexistent_binary_xyz", (), "ns_fail_test", None)
        mcp_client_module._shared_clients.pop(key, None)

        with self.assertRaises(Exception):
            mcp_client_module.get_shared_mcp_client(
                command="nonexistent_binary_xyz", args=[], namespace="ns_fail_test"
            )
        self.assertNotIn(key, mcp_client_module._shared_clients)

    def test_close_shared_mcp_client_is_safe_when_nothing_cached(self):
        from agent_mcp.client import close_shared_mcp_client
        # Should not raise even though no client was ever opened for this key.
        close_shared_mcp_client(command="echo", args=[], namespace="never_opened_ns")


class TestMCPServerResourcesAndPrompts(unittest.TestCase):
    """
    Real end-to-end checks against our own agent_mcp/server.py subprocess --
    no mocking of the MCP protocol -- since resources/prompts are new surface
    and the SDK's exact handler contract (str vs Iterable[ReadResourceContents],
    old-style vs *Result-style list handlers) is easy to get subtly wrong.
    """

    @classmethod
    def _run(cls, coro):
        import asyncio
        return asyncio.run(coro)

    @staticmethod
    async def _session():
        import sys
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command=sys.executable, args=["-m", "agent_mcp.server"])
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                yield session

    def test_lists_tool_category_and_skill_resources(self):
        async def run():
            async for session in self._session():
                result = await session.list_resources()
                uris = [str(r.uri) for r in result.resources]
                self.assertTrue(any(u.startswith("tool-category://") for u in uris))
                self.assertTrue(any(u.startswith("skill://") for u in uris))
                return uris
        uris = self._run(run())
        self.assertIn("tool-category://execution", uris)

    def test_reads_tool_category_resource_content(self):
        async def run():
            async for session in self._session():
                result = await session.read_resource("tool-category://execution")
                return result.contents[0].text
        text = self._run(run())
        self.assertIn("run_command", text)
        self.assertIn("run_tests", text)

    def test_reads_skill_resource_content(self):
        async def run():
            async for session in self._session():
                result = await session.read_resource("skill://fix_import_error")
                return result.contents[0].text
        text = self._run(run())
        self.assertIn("fix_import_error", text)

    def test_unknown_resource_raises(self):
        async def run():
            async for session in self._session():
                await session.read_resource("skill://does_not_exist")
        with self.assertRaises(Exception):
            self._run(run())

    def test_lists_prompts_from_skills(self):
        async def run():
            async for session in self._session():
                result = await session.list_prompts()
                return [p.name for p in result.prompts]
        names = self._run(run())
        self.assertIn("fix_import_error", names)

    def test_get_prompt_interpolates_task_argument(self):
        async def run():
            async for session in self._session():
                result = await session.get_prompt("fix_import_error", arguments={"task": "fix foo.py"})
                return result.messages[0].content.text
        text = self._run(run())
        self.assertIn("fix foo.py", text)
        self.assertIn("fix_import_error", text)


class TestMCPServerConfig(unittest.TestCase):

    def test_default_config_is_single_filesystem_server(self):
        from agent_mcp.config import load_mcp_server_configs
        # Run from an empty cwd with no env var set -> built-in default,
        # preserving today's behavior (one filesystem server) unchanged.
        import os
        old_env = os.environ.pop("MCP_SERVERS_CONFIG", None)
        old_cwd = os.getcwd()
        empty_dir = Path(tempfile.mkdtemp())
        try:
            os.chdir(empty_dir)
            configs = load_mcp_server_configs(path=None)
            self.assertEqual(len(configs), 1)
            self.assertEqual(configs[0].name, "mcp_fs")
            self.assertTrue(configs[0].enabled)
            self.assertTrue(configs[0].cwd_from_repo_root)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(empty_dir, ignore_errors=True)
            if old_env is not None:
                os.environ["MCP_SERVERS_CONFIG"] = old_env

    def test_loads_explicit_config_file(self):
        from agent_mcp.config import load_mcp_server_configs
        tmp = Path(tempfile.mkdtemp())
        try:
            cfg_path = tmp / "servers.json"
            cfg_path.write_text(json.dumps([
                {"name": "a", "command": "echo", "args": ["1"], "enabled": True},
                {"name": "b", "command": "echo", "args": ["2"], "enabled": False},
            ]))
            configs = load_mcp_server_configs(path=str(cfg_path))
            self.assertEqual(len(configs), 2)
            self.assertEqual(configs[0].namespace, "a")  # defaults to name when unset
            self.assertFalse(configs[1].enabled)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_explicit_config_raises(self):
        from agent_mcp.config import load_mcp_server_configs
        with self.assertRaises(FileNotFoundError):
            load_mcp_server_configs(path="/nonexistent/path/servers.json")

    def test_malformed_config_raises_value_error(self):
        from agent_mcp.config import load_mcp_server_configs
        tmp = Path(tempfile.mkdtemp())
        try:
            cfg_path = tmp / "bad.json"
            cfg_path.write_text("not json")
            with self.assertRaises(ValueError):
                load_mcp_server_configs(path=str(cfg_path))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_filter_tool_map_allow_list(self):
        from agent_mcp.config import filter_tool_map, MCPServerConfig
        cfg = MCPServerConfig(name="fs", command="x", namespace="fs", allowed_tools=["read_file"])
        tool_map = {"fs__read_file": 1, "fs__write_file": 2, "fs__delete_file": 3}
        filtered = filter_tool_map(tool_map, cfg)
        self.assertEqual(set(filtered.keys()), {"fs__read_file"})

    def test_filter_tool_map_deny_list_applies_without_allow_list(self):
        from agent_mcp.config import filter_tool_map, MCPServerConfig
        cfg = MCPServerConfig(name="fs", command="x", namespace="fs", denied_tools=[r"delete_.*", r"^move_"])
        tool_map = {"fs__read_file": 1, "fs__delete_file": 2, "fs__move_file": 3}
        filtered = filter_tool_map(tool_map, cfg)
        self.assertEqual(set(filtered.keys()), {"fs__read_file"})

    def test_filter_tool_map_allow_and_deny_combine(self):
        from agent_mcp.config import filter_tool_map, MCPServerConfig
        # denied_tools still applies even when the tool passed the allow-list --
        # deny always wins.
        cfg = MCPServerConfig(
            name="fs", command="x", namespace="fs",
            allowed_tools=["read_file", "delete_file"], denied_tools=[r"delete_.*"],
        )
        tool_map = {"fs__read_file": 1, "fs__delete_file": 2}
        filtered = filter_tool_map(tool_map, cfg)
        self.assertEqual(set(filtered.keys()), {"fs__read_file"})


# ─────────────────────────────────────────────────────────────────────────────
# 11. RAG Knowledge Base Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGKnowledgeBase(unittest.TestCase):

    def setUp(self):
        self.repo = make_repo({
            "main.py": "def main():\n    pass\n",
            "skills/my_skill.md": (
                "---\n"
                "name: my_skill\n"
                "trigger_keywords: [error, bug]\n"
                "summary: Fix a generic bug.\n"
                "---\n\nProcedure:\n1. Identify\n2. Fix\n"
            ),
        })
        # Write a minimal tools/specs.py stub in the repo
        (self.repo / "tools").mkdir(exist_ok=True)
        (self.repo / "tools" / "specs.py").write_text(
            'TOOL_SPECS = {\n'
            '    "read_file": {\n'
            '        "description": "Read a file.",\n'
            '        "when_to_use": ["When you know the path."],\n'
            '        "parameters": {"path": {"type": "string", "required": True}},\n'
            '        "output": "File contents.",\n'
            '    },\n'
            '}\n'
        )
        (self.repo / "tools" / "__init__.py").write_text("")

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_tool_spec_chunks_injected(self):
        from rag.indexer import RepoIndexer
        indexer = RepoIndexer(repo_root=str(self.repo))
        chunks = indexer._inject_tool_spec_chunks()
        self.assertGreater(len(chunks), 0)
        types = {c.chunk_type for c in chunks}
        self.assertIn("tool_summary", types)
        names = [c.symbol_name for c in chunks]
        self.assertIn("read_file", names)

    def test_skill_chunks_injected(self):
        from rag.indexer import RepoIndexer
        indexer = RepoIndexer(repo_root=str(self.repo))
        chunks = indexer._inject_skill_chunks()
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].chunk_type, "skill_summary")
        self.assertIn("my_skill", chunks[0].symbol_name or "")

    def test_tool_chunk_content(self):
        from rag.indexer import RepoIndexer
        indexer = RepoIndexer(repo_root=str(self.repo))
        chunks = indexer._inject_tool_spec_chunks()
        read_file_chunk = next(c for c in chunks if c.symbol_name == "read_file")
        self.assertIn("Read a file", read_file_chunk.content)
        self.assertIn("path", read_file_chunk.content)

    def test_skill_chunk_content(self):
        from rag.indexer import RepoIndexer
        indexer = RepoIndexer(repo_root=str(self.repo))
        chunks = indexer._inject_skill_chunks()
        self.assertIn("Fix a generic bug", chunks[0].content)
        self.assertIn("Procedure", chunks[0].content)

    def test_collect_chunks_includes_knowledge(self):
        from rag.indexer import RepoIndexer
        indexer = RepoIndexer(repo_root=str(self.repo))
        chunks = indexer._collect_chunks()
        chunk_types = {c.chunk_type for c in chunks}
        self.assertIn("tool_summary", chunk_types)
        self.assertIn("skill_summary", chunk_types)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Structured Logging Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoggingSetup(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="agent_logs_"))
        self._orig_log_dir = __import__("os").environ.get("AGENT_LOG_DIR")
        __import__("os").environ["AGENT_LOG_DIR"] = str(self.tmp)
        # Force re-configuration against the temp dir for this test.
        import core.logging_setup as logging_setup
        logging_setup._configured = False
        import logging as _logging
        _logging.getLogger(logging_setup._LOGGER_NAME).handlers.clear()

    def tearDown(self):
        import os as _os
        if self._orig_log_dir is None:
            _os.environ.pop("AGENT_LOG_DIR", None)
        else:
            _os.environ["AGENT_LOG_DIR"] = self._orig_log_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_log_event_writes_json_line(self):
        from core.logging_setup import log_event
        log_event("unit_test_event", run_id="abc123", tool="read_file", success=True)

        log_file = self.tmp / "agent-events.jsonl"
        self.assertTrue(log_file.exists())

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(lines[-1])
        self.assertEqual(record["event"], "unit_test_event")
        self.assertEqual(record["run_id"], "abc123")
        self.assertEqual(record["tool"], "read_file")
        self.assertTrue(record["success"])
        self.assertIn("ts", record)
        self.assertIn("level", record)

    def test_timed_context_manager_measures_duration(self):
        from core.logging_setup import timed
        with timed() as t:
            pass
        self.assertIsNotNone(t.duration_ms)
        self.assertGreaterEqual(t.duration_ms, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Federated Retrieval Tests (memory recall + cross-source rerank)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeEmbedder:
    """
    Deterministic, dependency-free stand-in for CodeEmbedder. Hashes words
    into a small fixed-size vector (a bag-of-words hashing trick) so texts
    sharing vocabulary score higher via cosine similarity -- without loading
    a real sentence-transformers model, keeping these tests fast.

    Uses zlib.crc32, not Python's built-in hash(): hash() on strings is
    salted with a random seed per process (PYTHONHASHSEED), so word->bucket
    assignments -- and therefore which test cases pass -- would silently
    vary from run to run despite this class's own docstring claiming
    "deterministic". crc32 is stable across processes.
    """
    DIM = 32

    def _vec(self, text: str):
        import numpy as np
        import zlib
        v = np.zeros(self.DIM, dtype="float32")
        for word in text.lower().split():
            v[zlib.crc32(word.encode()) % self.DIM] += 1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    def embed_texts(self, texts):
        import numpy as np
        if not texts:
            return np.empty((0, self.DIM), dtype="float32")
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query):
        import numpy as np
        return np.stack([self._vec(query)])


class TestMemoryRecallCandidates(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.memory = AgentMemory(persist_dir=str(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_memory_returns_no_candidates(self):
        results = self.memory.recall_candidates("anything", embedder=_FakeEmbedder())
        self.assertEqual(results, [])

    def test_ranks_by_similarity_to_query(self):
        self.memory.add_insight("the executor runs tools and steps")
        self.memory.add_insight("completely unrelated note about weather")
        results = self.memory.recall_candidates("executor tools", top_k=2, embedder=_FakeEmbedder())
        self.assertEqual(results[0]["content"], "the executor runs tools and steps")
        self.assertEqual(results[0]["source"], "memory")
        self.assertIn("chunk_id", results[0])

    def test_skips_inactive_items(self):
        self.memory.add_insight("a note")
        self.memory.long_term[0].status = "stale"
        results = self.memory.recall_candidates("note", embedder=_FakeEmbedder())
        self.assertEqual(results, [])


class TestApplyRelevanceFloor(unittest.TestCase):
    """RAGEngine.apply_relevance_floor(), tested without loading any model."""

    def test_passthrough_without_cross_encoder(self):
        from rag.retrieve import RAGEngine
        engine = RAGEngine.__new__(RAGEngine)  # skip __init__ -- no model loading needed
        engine._cross_encoder = None
        reranked = [{"rerank_score": -10.0}]
        result = engine.apply_relevance_floor(reranked, min_score=-4.0)
        self.assertEqual(result, reranked)

    def test_filters_below_threshold(self):
        from rag.retrieve import RAGEngine
        engine = RAGEngine.__new__(RAGEngine)
        engine._cross_encoder = object()  # any non-None sentinel
        reranked = [{"rerank_score": 2.0}, {"rerank_score": -10.0}]
        result = engine.apply_relevance_floor(reranked, min_score=-4.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rerank_score"], 2.0)

    def test_keeps_top1_flagged_when_all_below_threshold(self):
        from rag.retrieve import RAGEngine
        engine = RAGEngine.__new__(RAGEngine)
        engine._cross_encoder = object()
        reranked = [{"rerank_score": -10.0}, {"rerank_score": -20.0}]
        result = engine.apply_relevance_floor(reranked, min_score=-4.0)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["_low_relevance"])


class _StubRAGEngine:
    """Duck-typed RAGEngine stand-in for FederatedRetriever tests."""

    def __init__(self, code_results, cross_encoder=None):
        self._code_results = code_results
        self._cross_encoder = cross_encoder

    def hybrid_recall(self, query, vector_top_k, keyword_top_k):
        return [dict(r) for r in self._code_results]

    def rerank(self, query, candidates, top_k):
        out = []
        for i, c in enumerate(candidates):
            d = dict(c)
            d["rerank_score"] = len(candidates) - i  # preserve input order, descending
            out.append(d)
        return out[:top_k]

    def apply_relevance_floor(self, reranked, min_score):
        if self._cross_encoder is not None and min_score is not None and reranked:
            filtered = [r for r in reranked if r.get("rerank_score", 0.0) >= min_score]
            if filtered:
                return filtered
            return [dict(reranked[0], _low_relevance=True)]
        return reranked

    def format_context(self, results, max_chars_per_chunk=1400):
        return f"stub-context({len(results)} results)"


class _StubMemory:
    def __init__(self, candidates):
        self._candidates = candidates

    def recall_candidates(self, query, top_k=5):
        return [dict(c) for c in self._candidates[:top_k]]


class TestFederatedRetriever(unittest.TestCase):

    def test_merges_and_tags_sources(self):
        from rag.federated import FederatedRetriever
        code = [{"chunk_id": "c1", "content": "code chunk"}]
        mem = [{"chunk_id": "m1", "content": "memory chunk"}]
        retriever = FederatedRetriever(rag_engine=_StubRAGEngine(code), memory=_StubMemory(mem))

        results = retriever.retrieve("query", top_k=5, min_score=None)

        source_types = {r["source_type"] for r in results}
        self.assertEqual(source_types, {"repo", "memory"})

    def test_no_memory_source_when_memory_is_none(self):
        from rag.federated import FederatedRetriever
        code = [{"chunk_id": "c1", "content": "code chunk"}]
        retriever = FederatedRetriever(rag_engine=_StubRAGEngine(code), memory=None)

        results = retriever.retrieve("query", min_score=None)

        self.assertTrue(all(r["source_type"] == "repo" for r in results))

    def test_empty_when_no_candidates(self):
        from rag.federated import FederatedRetriever
        retriever = FederatedRetriever(rag_engine=_StubRAGEngine([]), memory=_StubMemory([]))

        self.assertEqual(retriever.retrieve("query"), [])

    def test_applies_relevance_floor_across_merged_pool(self):
        from rag.federated import FederatedRetriever
        code = [{"chunk_id": "c1", "content": "irrelevant"}]
        mem = [{"chunk_id": "m1", "content": "also irrelevant"}]
        # rerank_score for these two candidates will be 2 and 1 (stub assigns
        # descending scores by input order); min_score=1.5 should drop the
        # second one but keep the merged pool's ordering intact.
        retriever = FederatedRetriever(
            rag_engine=_StubRAGEngine(code, cross_encoder=object()), memory=_StubMemory(mem)
        )

        results = retriever.retrieve("query", min_score=1.5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_type"], "repo")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
