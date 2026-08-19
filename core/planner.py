from __future__ import annotations

from typing import Any, List, Optional
import json
import re

from .state import AgentState, PlanStep, StepStatus
from tools.registry import ToolRegistry


class Planner:
    """
    Planner decomposes the user's request into executable coding-agent steps.

    Supports:
    1. LLM planner mode:
       - Uses client.chat(...)
       - Asks LLM to return structured JSON
       - Each step includes task, reason, expected_output, suggested_tools

    2. Fallback planner mode:
       - No LLM needed
       - Uses deterministic rule-based plans
       - Useful for stable demos and debugging
    """

    def __init__(self, client: Optional[Any] = None, tool_registry: Optional[ToolRegistry] = None):
        self.client = client
        # Optional: when given, the LLM prompt shows a compact category-level
        # tool summary (tools/registry.py) instead of a hardcoded flat tool
        # name list -- keeps prompt size roughly constant regardless of how
        # many built-in + MCP tools end up registered. None preserves the
        # old hardcoded-list behavior (e.g. fallback-mode callers that never
        # touch the LLM prompt at all).
        self.tool_registry = tool_registry

    def create_initial_plan(self, state: AgentState) -> List[PlanStep]:
        """
        Create the initial plan for the user request.
        """
        if self.client is None:
            steps = self._fallback_plan(state.input_query)
            state.add_plan(steps)
            state.add_history(
                "Planner created fallback initial plan:\n"
                f"{state.plan_summary(verbose=True)}"
            )
            return steps

        prompt = self._build_initial_plan_prompt(state)

        response = self.client.chat(
            [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ]
        )

        content = self._normalize_llm_response(response)
        steps = self._parse_plan(content)

        if not steps:
            steps = self._fallback_plan(state.input_query)
            state.add_history(
                "Planner failed to parse LLM plan, used fallback plan instead."
            )
        else:
            state.add_history("Planner created LLM initial plan from model response.")

        state.add_plan(steps)
        state.add_history(
            "Initial plan:\n"
            f"{state.plan_summary(verbose=True)}"
        )

        return steps

    def adjust_plan(self, state: AgentState) -> List[PlanStep]:
        """
        Re-plan after tool failure, validation failure, or incomplete result.
        """
        state.replan_count += 1

        if self.client is None:
            new_steps = self._fallback_replan(state)

            next_id = max([s.step_id for s in state.plan], default=0) + 1
            for i, step in enumerate(new_steps):
                step.step_id = next_id + i

            state.plan.extend(new_steps)
            state.add_history(
                "Planner created fallback recovery plan:\n"
                f"{state.plan_summary(verbose=True)}"
            )
            return state.plan

        prompt = self._build_replan_prompt(state)

        response = self.client.chat(
            [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ]
        )

        content = self._normalize_llm_response(response)
        new_steps = self._parse_plan(content)

        if not new_steps:
            new_steps = self._fallback_replan(state)
            state.add_history(
                "Planner failed to parse LLM recovery plan, used fallback recovery plan."
            )
        else:
            state.add_history("Planner created LLM recovery plan from model response.")

        next_id = max([s.step_id for s in state.plan], default=0) + 1
        for i, step in enumerate(new_steps):
            step.step_id = next_id + i

        state.plan.extend(new_steps)
        state.add_history(
            "Updated plan after replanning:\n"
            f"{state.plan_summary(verbose=True)}"
        )

        return state.plan

    def _system_prompt(self) -> str:
        tools_note = (
            "You will be shown available tools as CATEGORIES, not individual tool "
            "descriptions -- there may be many tools (including ones from MCP servers) "
            "and listing each one would blow up this prompt. For suggested_tools, either "
            "name a category directly (e.g. \"mutation\") to let the Executor pick within "
            "it, or name a specific tool if you already know it (e.g. \"read_file\") -- "
            "the Executor expands either into the concrete tools it needs."
            if self.tool_registry else
            "Use only available tool names: list_files, search_code, retrieve_context, "
            "read_file, write_file, replace_in_file, apply_patch, run_command, run_tests, "
            "identify_error, git_diff."
        )

        return f"""
You are a senior planner for a general-purpose coding agent -- not limited to bug fixes.
Tasks you may be asked to plan include: diagnosing and fixing a bug, implementing a new
feature, refactoring existing code, writing tests, upgrading a dependency, reviewing a
diff, or just analyzing/summarizing a repository.

Your job:
- Break the user request into concrete executable steps.
- Each step should be something the Executor can complete using tools.
- Prefer inspecting the repository before making conclusions.
- Prefer retrieve_context for architecture, project summary, missing-parts analysis, and broad semantic questions.
- Prefer search_code for exact symbols, class names, function names, imports, and error messages.
- Prefer read_file when the target file path is already known.
- Add validation steps (tests, syntax checks) whenever the task changes code, not just for bug fixes --
  a new feature or a refactor needs to be verified too, just not necessarily against a pre-existing failing test.
- Do not add edit/apply_patch steps for pure analysis, summarization, or review-only tasks.
- {tools_note}

Return only valid JSON.

Schema:
[
  {{
    "task": "Clear task description",
    "reason": "Why this step is needed",
    "expected_output": "What should be produced by this step",
    "suggested_tools": ["list_files", "search_code", "read_file"]
  }}
]
""".strip()

    def _top_level_listing(self, repo_root: str) -> str:
        """Quick top-level file listing so the planner knows what exists."""
        from pathlib import Path as _Path
        skip = {".git", "__pycache__", ".venv", "venv", "node_modules", ".agent_index"}
        lines = []
        try:
            root = _Path(repo_root)
            for entry in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if entry.name.startswith(".") or entry.name in skip:
                    continue
                suffix = "/" if entry.is_dir() else ""
                lines.append(f"  {entry.name}{suffix}")
                if entry.is_dir():
                    try:
                        for child in sorted(entry.iterdir(), key=lambda p: p.name.lower())[:8]:
                            if child.name.startswith(".") or child.name in skip:
                                continue
                            lines.append(f"    {child.name}{'/' if child.is_dir() else ''}")
                    except PermissionError:
                        pass
        except Exception:
            pass
        return "\n".join(lines) or "(empty)"

    def _build_initial_plan_prompt(self, state: AgentState) -> str:
        file_listing = self._top_level_listing(state.repo_root)
        return f"""
User request:
{state.input_query}

Repository top-level layout (use this to plan which files to read):
{file_listing}

Create a short execution plan for a coding agent.

Rules:
1. If the request is analysis/review/summary only, do not include editing steps.
2. If the request asks to fix a bug, implement a feature, refactor, or write tests,
   include read/search steps, the actual change, and a validation step.
3. Each step must include:
   - task
   - reason
   - expected_output
   - suggested_tools
4. {self._available_tools_prompt_block()}
5. Return only JSON.
""".strip()

    def _available_tools_prompt_block(self) -> str:
        """
        Rule #4 of the initial-plan prompt: what tools exist. Category-level
        (compact, scales with any number of MCP tools) when a ToolRegistry
        was given; the old hardcoded flat list otherwise -- see
        Planner.__init__ and tools/registry.py.
        """
        if self.tool_registry:
            return (
                "Available tool categories (name a category, or a specific tool "
                "if you already know its exact name):\n"
                f"{self.tool_registry.category_summary_prompt()}"
            )
        return (
            "Use only available tool names: list_files, search_code, retrieve_context, "
            "read_file, write_file, replace_in_file, apply_patch, run_command, run_tests, "
            "identify_error, git_diff."
        )

    def _build_replan_prompt(self, state: AgentState) -> str:
        errors_deduped = list(dict.fromkeys(state.errors_seen))[-5:]
        errors_text = "\n".join(f"- {e[:300]}" for e in errors_deduped) or "None"

        return f"""
The coding agent needs to re-plan.

Original user request:
{state.input_query}

Current plan:
{state.plan_summary(verbose=True)}

Recent tool results:
{state.recent_tool_summary(limit=5)}

Errors seen:
{errors_text}

Files read:
{state.files_read}

Files modified:
{state.files_modified}

Create a short recovery plan.

Rules:
1. Focus only on unresolved or failed parts.
2. Use error logs and recent tool results to identify the next action.
3. Each step must include:
   - task
   - reason
   - expected_output
   - suggested_tools
4. Do not repeat completed work unless necessary.
5. Return only JSON.
""".strip()

    def _normalize_llm_response(self, response: Any) -> str:
        """
        Make Planner compatible with different LLM clients.
        """
        if isinstance(response, str):
            return response

        if isinstance(response, dict):
            if "content" in response:
                return str(response["content"])

            if "message" in response and isinstance(response["message"], dict):
                return str(response["message"].get("content", ""))

            if "choices" in response:
                try:
                    return response["choices"][0]["message"]["content"]
                except Exception:
                    return str(response)

        return str(response)

    def _parse_plan(self, content: str) -> List[PlanStep]:
        """
        Parse LLM JSON plan into PlanStep objects.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = self._extract_json_array(content)

        if not isinstance(data, list):
            return []

        steps: List[PlanStep] = []

        for idx, item in enumerate(data, start=1):
            if isinstance(item, str):
                steps.append(
                    PlanStep(
                        step_id=idx,
                        task=item,
                        expected_output="Complete this step.",
                        suggested_tools=[],
                        status=StepStatus.PENDING,
                        planner_notes=(
                            "Planner generated this string step from the LLM response. "
                            "No explicit reason was provided."
                        ),
                    )
                )
                continue

            if not isinstance(item, dict):
                continue

            task = item.get("task") or item.get("description")
            if not task:
                continue

            suggested_tools = item.get("suggested_tools", [])
            if isinstance(suggested_tools, str):
                suggested_tools = [suggested_tools]

            reason = (
                item.get("reason")
                or item.get("planner_notes")
                or item.get("rationale")
                or "Planner generated this step based on the user request."
            )

            steps.append(
                PlanStep(
                    step_id=idx,
                    task=task,
                    expected_output=item.get("expected_output", ""),
                    suggested_tools=suggested_tools,
                    status=StepStatus.PENDING,
                    planner_notes=reason,
                )
            )

        return steps

    def _extract_json_array(self, text: str) -> Any:
        """
        Extract JSON array if LLM returned extra text around the JSON.
        """
        if not text:
            return None

        cleaned = text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    # Keyword sets for the deterministic (no-LLM) planner's task classification.
    # Checked in this order (first match wins) since some keywords overlap
    # across categories (e.g. "update" alone is too generic to mean
    # dependency_upgrade specifically) -- more specific phrasings are listed
    # for the categories checked first.
    _TASK_KEYWORDS = {
        "test_writing": [
            "write test", "add test", "add unit test", "write unit test",
            "test coverage", "add coverage", "write a test",
        ],
        "dependency_upgrade": [
            "upgrade", "bump version", "bump the version", "update dependency",
            "update dependencies", "migrate to", "update to version",
        ],
        "refactor": [
            "refactor", "clean up", "clean this up", "simplify", "restructure",
            "reorganize", "extract a function", "extract method", "de-duplicate", "dedupe",
        ],
        "feature_add": [
            "implement a", "implement the", "add a feature", "add feature", "new feature",
            "add support for", "create a new", "build a", "add an endpoint", "add a command",
        ],
        "review": [
            "review this diff", "code review", "review the changes", "review my changes", "audit the diff",
        ],
        "analysis": [
            "analyze", "inspect", "summarize", "summary", "explain", "identify",
            "review", "what is implemented", "what is missing", "grounded bullet",
            "project structure",
        ],
        "bug_fix": [
            "fix", "bug", "debug", "broken", "crash", "solve", "error",
            "modify", "change", "update", "patch", "add code", "complete code",
        ],
    }

    def _classify_task(self, query_lower: str) -> str:
        for category, keywords in self._TASK_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return category
        return "bug_fix"  # default: assume some form of code change is wanted

    def _fallback_plan(self, user_query: str) -> List[PlanStep]:
        """
        Deterministic default plan -- routes to a category-specific template
        (see _TASK_KEYWORDS) instead of the old binary analysis/code-change
        split, so a general-purpose request (write tests, refactor, upgrade a
        dependency, review a diff) gets a plan shaped for that task instead
        of the generic "read then patch" template.
        """
        query_lower = user_query.lower()
        category = self._classify_task(query_lower)

        builders = {
            "analysis": self._plan_analysis,
            "review": self._plan_review,
            "test_writing": lambda: self._plan_test_writing(user_query),
            "refactor": lambda: self._plan_refactor(user_query),
            "feature_add": lambda: self._plan_feature_add(user_query),
            "dependency_upgrade": lambda: self._plan_dependency_upgrade(user_query),
            "bug_fix": lambda: self._plan_bug_fix(user_query),
        }
        return builders[category]()

    def _plan_analysis(self) -> List[PlanStep]:
        return [
            PlanStep(
                step_id=1,
                task="Inspect repository structure",
                expected_output="A concise overview of the repository layout.",
                suggested_tools=["list_files"],
                planner_notes=(
                    "Planner starts by listing files because the agent needs to understand "
                    "the repository layout before deciding which modules are relevant."
                ),
            ),
            PlanStep(
                step_id=2,
                task="Search for core executor, planner, tools, and rag modules",
                expected_output=(
                    "Relevant files and symbols related to planning, execution, tools, "
                    "memory, and retrieval."
                ),
                suggested_tools=["search_code", "retrieve_context"],
                planner_notes=(
                    "Planner searches for core modules so the Executor can locate the files "
                    "responsible for the coding-agent architecture."
                ),
            ),
            PlanStep(
                step_id=3,
                task="Read relevant files to understand their contents",
                expected_output=(
                    "Concrete observations about executor, planner, state, memory, tools, "
                    "and RAG implementation."
                ),
                suggested_tools=["read_file", "retrieve_context"],
                planner_notes=(
                    "Planner asks Executor to read or retrieve context because summaries "
                    "should be grounded in actual code, not only file names."
                ),
            ),
            PlanStep(
                step_id=4,
                task="Summarize the coding agent project in 3 grounded bullet points",
                expected_output=(
                    "A concise project summary explaining what is implemented and what is missing."
                ),
                suggested_tools=["retrieve_context", "read_file"],
                planner_notes=(
                    "Planner ends with a grounded summary after repository structure and relevant "
                    "code have been inspected."
                ),
            ),
        ]

    def _plan_review(self) -> List[PlanStep]:
        return [
            PlanStep(
                step_id=1,
                task="Show the current diff to see what changed",
                expected_output="The full set of pending changes.",
                suggested_tools=["git_diff"],
                planner_notes="Review starts from the diff, not the whole repo -- that's the actual review surface.",
            ),
            PlanStep(
                step_id=2,
                task="Read the files touched by the diff for surrounding context",
                expected_output="Enough context to judge whether each change is correct and consistent with the rest of the file.",
                suggested_tools=["read_file", "search_code"],
                planner_notes="A diff line rarely makes sense without the function/file it sits in.",
            ),
            PlanStep(
                step_id=3,
                task="Identify correctness issues, edge cases, or inconsistencies introduced by the diff",
                expected_output="A list of concrete findings, each tied to a file and line.",
                suggested_tools=["retrieve_context", "search_code"],
                planner_notes="Findings must cite the actual diff/file, not general best-practice guesses.",
            ),
            PlanStep(
                step_id=4,
                task="Summarize the review: what's solid, what needs a second look, nothing invented",
                expected_output="A grounded review summary.",
                suggested_tools=["retrieve_context"],
                planner_notes="Review tasks never edit code -- no apply_patch/write_file step here.",
            ),
        ]

    def _plan_test_writing(self, user_query: str) -> List[PlanStep]:
        return [
            PlanStep(
                step_id=1,
                task="Inspect repository structure and locate the existing test suite layout/conventions",
                expected_output="Where tests live and what pattern/framework they follow.",
                suggested_tools=["list_files", "search_code"],
                planner_notes="New tests should match the project's existing conventions, not invent a new style.",
            ),
            PlanStep(
                step_id=2,
                task=f"Read the code that needs test coverage: {user_query}",
                expected_output="Concrete understanding of the function/module's behavior, including edge cases.",
                suggested_tools=["read_file", "retrieve_context"],
                planner_notes="Tests grounded in actual behavior, not assumed behavior.",
            ),
            PlanStep(
                step_id=3,
                task="Write the test cases (happy path + edge cases) following the existing test file conventions",
                expected_output="New or updated test file(s).",
                suggested_tools=["write_file", "replace_in_file"],
                planner_notes="A dedicated test-writing step, not folded into a generic 'apply a patch' step.",
            ),
            PlanStep(
                step_id=4,
                task="Run the test suite to confirm the new tests pass (and don't break existing ones)",
                expected_output="Passing tests, or a clear remaining failure to fix.",
                suggested_tools=["run_tests"],
                planner_notes="A test that was never run is not verified.",
            ),
        ]

    def _plan_refactor(self, user_query: str) -> List[PlanStep]:
        return [
            PlanStep(
                step_id=1,
                task=f"Find every usage/call site relevant to the refactor: {user_query}",
                expected_output="A complete list of files/call sites the refactor will touch.",
                suggested_tools=["search_code", "retrieve_context"],
                planner_notes="A refactor that misses a call site breaks the build -- search broadly before touching anything.",
            ),
            PlanStep(
                step_id=2,
                task="Read the current implementation and every call site found above",
                expected_output="Full understanding of current behavior that must be preserved.",
                suggested_tools=["read_file"],
                planner_notes="Refactors must preserve behavior; read before rewriting.",
            ),
            PlanStep(
                step_id=3,
                task="Run the existing tests to capture a pre-refactor baseline",
                expected_output="A baseline pass/fail snapshot to compare against after the change.",
                suggested_tools=["run_tests"],
                planner_notes="Without a baseline there's no way to tell if the refactor changed behavior.",
            ),
            PlanStep(
                step_id=4,
                task="Apply the refactor across all identified call sites",
                expected_output="Updated code with the same external behavior.",
                suggested_tools=["apply_patch", "replace_in_file"],
                planner_notes="Applied only after every call site is known, to avoid a half-finished rename/extraction.",
            ),
            PlanStep(
                step_id=5,
                task="Run the test suite again and confirm it matches the pre-refactor baseline",
                expected_output="Same (or better) test results as the baseline -- no behavior change.",
                suggested_tools=["run_tests"],
                planner_notes="A refactor is only successful if behavior didn't change.",
            ),
        ]

    def _plan_feature_add(self, user_query: str) -> List[PlanStep]:
        return [
            PlanStep(
                step_id=1,
                task="Inspect repository structure to see where a new feature like this would live",
                expected_output="The relevant module/directory and existing patterns to follow.",
                suggested_tools=["list_files", "retrieve_context"],
                planner_notes="New code should follow the project's existing structure, not invent a new one.",
            ),
            PlanStep(
                step_id=2,
                task=f"Find similar existing functionality to model the new feature on: {user_query}",
                expected_output="An example of how a comparable feature is implemented in this codebase.",
                suggested_tools=["search_code", "retrieve_context"],
                planner_notes="Prefer extending established patterns over introducing a new one.",
            ),
            PlanStep(
                step_id=3,
                task="Read the specific files that will need to change or be created",
                expected_output="Concrete understanding of what needs to be added and where it plugs in.",
                suggested_tools=["read_file"],
                planner_notes="Grounded in the actual integration points, not assumed ones.",
            ),
            PlanStep(
                step_id=4,
                task="Implement the feature",
                expected_output="New/changed code implementing the requested feature.",
                suggested_tools=["write_file", "apply_patch", "replace_in_file"],
                planner_notes="A feature is new code, not just a patch to existing logic -- write_file is often needed alongside apply_patch.",
            ),
            PlanStep(
                step_id=5,
                task="Run tests (and add new ones if the feature has no coverage) to validate the implementation",
                expected_output="Passing tests covering the new feature, or a clear remaining error.",
                suggested_tools=["run_tests"],
                planner_notes="A feature without any test run is unverified, same bar as a bug fix.",
            ),
        ]

    def _plan_dependency_upgrade(self, user_query: str) -> List[PlanStep]:
        return [
            PlanStep(
                step_id=1,
                task="Locate dependency manifests (requirements.txt, package.json, pyproject.toml, etc.)",
                expected_output="Every file that declares the dependency being upgraded.",
                suggested_tools=["list_files", "search_code"],
                planner_notes="A dependency can be pinned in more than one manifest -- check both, not just the obvious one.",
            ),
            PlanStep(
                step_id=2,
                task=f"Find all usages of the dependency in the codebase to spot breaking-change risk: {user_query}",
                expected_output="Every call site that uses the dependency's API.",
                suggested_tools=["search_code", "retrieve_context"],
                planner_notes="An upgrade can silently break a call site using a removed/changed API -- find them before bumping the version.",
            ),
            PlanStep(
                step_id=3,
                task="Update the version pin(s) in the manifest(s)",
                expected_output="Updated manifest file(s).",
                suggested_tools=["replace_in_file"],
                planner_notes="A small, targeted edit -- not a rewrite of the manifest.",
            ),
            PlanStep(
                step_id=4,
                task="Run the test suite to catch any breaking changes from the upgrade",
                expected_output="Passing tests, or specific failures pointing at what the new version broke.",
                suggested_tools=["run_tests", "run_command"],
                planner_notes="The only reliable way to know if an upgrade is safe is to actually run the tests against it.",
            ),
        ]

    def _plan_bug_fix(self, user_query: str) -> List[PlanStep]:
        return [
            PlanStep(
                step_id=1,
                task="Inspect repository structure",
                expected_output="A concise overview of the repository layout.",
                suggested_tools=["list_files"],
                planner_notes=(
                    "Planner starts by listing files to understand the codebase before making changes."
                ),
            ),
            PlanStep(
                step_id=2,
                task=f"Search the codebase for files relevant to the user request: {user_query}",
                expected_output="A list of relevant files, symbols, or code locations.",
                suggested_tools=["search_code", "retrieve_context"],
                planner_notes=(
                    "Planner searches for relevant code locations before reading or editing files."
                ),
            ),
            PlanStep(
                step_id=3,
                task="Read the most relevant files and identify the implementation gap",
                expected_output="A diagnosis of what needs to be changed.",
                suggested_tools=["read_file", "retrieve_context"],
                planner_notes=(
                    "Planner asks Executor to inspect the relevant files before modifying code."
                ),
            ),
            PlanStep(
                step_id=4,
                task="Apply a minimal code change to address the issue",
                expected_output="A focused patch that changes only the necessary files.",
                suggested_tools=["apply_patch", "replace_in_file"],
                planner_notes=(
                    "Planner includes an edit step because the user request appears to require implementation or fixing."
                ),
            ),
            PlanStep(
                step_id=5,
                task="Run validation using tests or syntax checks",
                expected_output="Passing tests or clear remaining error output.",
                suggested_tools=["run_tests", "run_command"],
                planner_notes=(
                    "Planner includes validation to confirm the code change works and did not break the project."
                ),
            ),
        ]

    def _fallback_replan(self, state: AgentState) -> List[PlanStep]:
        """
        Deterministic recovery plan after failure.
        """
        return [
            PlanStep(
                step_id=1,
                task="Analyze the latest error or failed tool result",
                expected_output="Root cause of the failure.",
                suggested_tools=["identify_error", "search_code"],
                planner_notes=(
                    "Planner starts recovery by analyzing the latest error before attempting another fix."
                ),
            ),
            PlanStep(
                step_id=2,
                task="Read the files most likely related to the failure",
                expected_output="Relevant code context for the failure.",
                suggested_tools=["read_file", "retrieve_context"],
                planner_notes=(
                    "Planner asks Executor to inspect related files so the next change is grounded."
                ),
            ),
            PlanStep(
                step_id=3,
                task="Apply a small corrective patch",
                expected_output="A minimal patch that addresses the root cause.",
                suggested_tools=["apply_patch", "replace_in_file"],
                planner_notes=(
                    "Planner chooses a small corrective patch to avoid unrelated rewrites."
                ),
            ),
            PlanStep(
                step_id=4,
                task="Run validation again",
                expected_output="Passing tests or a clear remaining error.",
                suggested_tools=["run_tests", "run_command"],
                planner_notes=(
                    "Planner validates again after the corrective patch to check whether recovery succeeded."
                ),
            ),
        ]