from __future__ import annotations

from typing import Any, List, Optional
import json
import re

from .state import AgentState, PlanStep, StepStatus


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

    def __init__(self, client: Optional[Any] = None):
        self.client = client

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
        return """
You are a senior planner for a coding agent.

Your job:
- Break the user request into concrete executable steps.
- Each step should be something the Executor can complete using tools.
- Prefer inspecting the repository before making conclusions.
- Prefer retrieve_context for architecture, project summary, missing-parts analysis, and broad semantic questions.
- Prefer search_code for exact symbols, class names, function names, imports, and error messages.
- Prefer read_file when the target file path is already known.
- Add validation steps only if the task involves code modification or testing.
- Do not add edit/apply_patch steps for pure analysis or summarization tasks.

Return only valid JSON.

Schema:
[
  {
    "task": "Clear task description",
    "reason": "Why this step is needed",
    "expected_output": "What should be produced by this step",
    "suggested_tools": ["list_files", "search_code", "read_file"]
  }
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
2. If the request asks to fix or implement code, include read/search, edit, and validation steps.
3. Each step must include:
   - task
   - reason
   - expected_output
   - suggested_tools
4. Use only available tool names:
   - list_files
   - search_code
   - retrieve_context
   - read_file
   - write_file
   - replace_in_file
   - apply_patch
   - run_command
   - run_tests
   - identify_error
   - git_diff
5. Return only JSON.
""".strip()

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

    def _fallback_plan(self, user_query: str) -> List[PlanStep]:
        """
        Deterministic default plan.

        If the task is analysis/summarization, do not edit files.
        If the task is implementation/fix, include patch and validation.
        """
        query_lower = user_query.lower()

        is_analysis_task = any(
            keyword in query_lower
            for keyword in [
                "analyze",
                "inspect",
                "summarize",
                "summary",
                "explain",
                "identify",
                "review",
                "what is implemented",
                "what is missing",
                "grounded bullet",
                "project structure",
            ]
        )

        is_code_change_task = any(
            keyword in query_lower
            for keyword in [
                "fix",
                "implement",
                "modify",
                "change",
                "update",
                "patch",
                "debug",
                "solve",
                "add code",
                "complete code",
            ]
        )

        if is_analysis_task and not is_code_change_task:
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