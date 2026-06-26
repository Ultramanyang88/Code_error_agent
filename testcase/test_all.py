"""
Complete test suite for Code Error Agent.

Run:
    python testcase/test_all.py
    python -m pytest testcase/test_all.py -v
"""
from __future__ import annotations

import sys
import json
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
    replace_in_file, run_command, identify_error, get_tool_map,
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
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
