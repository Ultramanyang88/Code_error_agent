from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import re
import traceback

from .state import AgentState, PlanStep, RunStatus, StepStatus, ToolResult
from tools.specs import TOOL_SPECS
from skills.registry import SkillRegistry
from .memory import AgentMemory


class Executor:
    """
    Executor runs one plan step at a time.

    Main responsibilities:
    - Build context from state, memory, and RAG.
    - Ask LLM which tool to call.
    - Execute the selected tool.
    - Store tool results.
    - Mark step completed or failed.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        tools: Optional[Dict[str, Any]] = None,
        memory: Optional[AgentMemory] = None,
        max_tool_rounds: int = 4,
        skills_dir: str= "skills"
    ):
        self.client = client
        self.tools = tools or {}
        self.memory = memory or AgentMemory()
        self.max_tool_rounds = max_tool_rounds
        self.skill_registry = SkillRegistry(skills_dir=skills_dir)
    
    def _validate_tool_arg(
            self,
            tool_name: str,
            arguments: Dict[str, Any],
    ) -> Optional[str]:
        """return an error string if required args are missing, else None."""
        spec = TOOL_SPECS.get(tool_name)
        if not spec:
            return None

        params = spec.get("parameters", {})
        missing = [
            name
            for name, meta in params.items()
            if meta.get("required") and arguments.get(name) is None
        ]
        if missing:
            return f"Missing required args for {tool_name}: {missing}"
        return None

    def execute_current_step(self, state: AgentState) -> AgentState:
        current_step = state.get_current_step()

        if current_step is None:
            state.run_status = RunStatus.COMPLETED
            state.final_answer = self._build_final_answer(state)
            return state

        current_step.mark_running()
        print(f"\n  [step {current_step.step_id}] {current_step.task}")
        if current_step.suggested_tools:
            print(f"  [tools]  {', '.join(current_step.suggested_tools)}")

        try:
            if self.client is None:
                result_text = self._fallback_execute_step(current_step, state)
            else:
                result_text = self._llm_execute_step(current_step, state)

            if current_step.tool_results and not any(r.success for r in current_step.tool_results):
                error = current_step.tool_results[-1].error or "All tools failed for this step."
                current_step.mark_failed(error)
                print(f"  [step {current_step.step_id}] FAILED — {error[:120]}")
                state.add_history(
                    f"Step {current_step.step_id} failed: {current_step.task}\n{error}"
                )
            else:
                current_step.mark_completed(result_text)
                print(f"  [step {current_step.step_id}] DONE")
                state.add_history(
                    f"Step {current_step.step_id} completed: {current_step.task}\n{result_text}"
                )

        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}"
            current_step.mark_failed(error)
            state.errors_seen.append(error)
            state.add_history(
                f"Step {current_step.step_id} failed: {current_step.task}\n{error}"
            )

        return state

    def run_step(self, step: PlanStep, state: AgentState) -> ToolResult:
        """
        Compatibility wrapper if main.py calls executor.run_step(step, state).
        """
        step.mark_running()

        try:
            if self.client is None:
                output = self._fallback_execute_step(step, state)
            else:
                output = self._llm_execute_step(step, state)

            if step.tool_results and not any(r.success for r in step.tool_results):
                error = step.tool_results[-1].error or "All tools failed for this step."
                step.mark_failed(error)

                result = ToolResult(
                    tool_name="executor",
                    success=False,
                    output="",
                    error=error,
                    metadata={"step_id": step.step_id, "task": step.task},
                )
            else:
                step.mark_completed(output)

                result = ToolResult(
                    tool_name="executor",
                    success=True,
                    output=output,
                    metadata={"step_id": step.step_id, "task": step.task},
                )

        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)}"
            step.mark_failed(error)

            result = ToolResult(
                tool_name="executor",
                success=False,
                output="",
                error=error,
                metadata={"step_id": step.step_id, "task": step.task},
            )

        state.add_tool_result(result)
        self.memory.update_from_tool_result(result)
        return result

    def _llm_execute_step(self, step: PlanStep, state: AgentState) -> str:
        messages = self._build_messages(step, state)
        final_outputs: List[str] = []

        for round_idx in range(self.max_tool_rounds):
            response = self.client.chat(messages)
            content = self._normalize_llm_response(response)

            tool_call = self._parse_tool_call(content)

            # No tool call means the LLM has produced the final answer for this step.
            if tool_call is None:
                final_outputs.append(content)
                break

            tool_name = tool_call.get("tool_name") or ""
            arguments = tool_call.get("arguments", {})

            tool_result = self._execute_tool(tool_name, arguments, state)

            step.tool_results.append(tool_result)
            state.add_tool_result(tool_result)
            self.memory.update_from_tool_result(tool_result)

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

            # Use role=user instead of role=tool for better compatibility with Ollama/local models.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool execution result:\n"
                        f"{tool_result.to_text(max_chars=2500)}\n\n"
                        "Based on this result, either call the next tool using JSON only, "
                        "or return the final concise answer for the current step."
                    ),
                }
            )

            final_outputs.append(tool_result.to_text(max_chars=1500))

            if self._tool_result_is_enough(step, tool_result):
                break

        if not final_outputs:
            return "No output produced."

        return "\n\n".join(final_outputs)

    def _fallback_execute_step(self, step: PlanStep, state: AgentState) -> str:
        """
        Rule-based execution before the LLM tool-calling layer is stable.
        """
        outputs: List[str] = []

        tools_to_try = step.suggested_tools or []

        if not tools_to_try:
            task_lower = step.task.lower()

            if "structure" in task_lower or "layout" in task_lower or "list" in task_lower:
                tools_to_try = ["list_files"]
            elif "search" in task_lower:
                tools_to_try = ["search_code"]
            elif "read" in task_lower or "inspect" in task_lower:
                tools_to_try = ["read_file"]
            elif "test" in task_lower or "validation" in task_lower:
                tools_to_try = ["run_tests"]
            elif "summarize" in task_lower or "summary" in task_lower or "analyze" in task_lower:
                tools_to_try = ["retrieve_context"]
            else:
                tools_to_try = ["retrieve_context"]

        for tool_name in tools_to_try:
            if tool_name not in self.tools:
                outputs.append(f"Tool {tool_name} is not available.")
                continue

            arguments = self._build_fallback_arguments(tool_name, step, state)
            tool_result = self._execute_tool(tool_name, arguments, state)

            step.tool_results.append(tool_result)
            state.add_tool_result(tool_result)
            self.memory.update_from_tool_result(tool_result)

            outputs.append(tool_result.to_text())

            if tool_result.success:
                break

        return "\n\n".join(outputs)

    def _build_fallback_arguments(
        self,
        tool_name: str,
        step: PlanStep,
        state: AgentState,
    ) -> Dict[str, Any]:
        if tool_name == "list_files":
            return {
                "directory": ".",
                "max_depth": 3,
                "include_hidden": False,
            }

        if tool_name == "search_code":
            return {
                "query": state.input_query,
                "include_glob": "*.py",
                "max_results": 50,
                "use_regex": False,
            }

        if tool_name == "retrieve_context":
            return {
                "query": f"{state.input_query}\nCurrent step: {step.task}",
                "top_k": 6,
                "force_rebuild": False,
            }

        if tool_name == "run_tests":
            return {
                "test_path": None,
                "timeout": 60,
            }

        if tool_name == "run_command":
            return {
                "command": "python -m compileall .",
                "timeout": 60,
            }

        if tool_name == "read_file":
            task_lower = step.task.lower()

            if "executor" in task_lower:
                path = "core/executor.py"
            elif "planner" in task_lower:
                path = "core/planner.py"
            elif "tools" in task_lower:
                path = "tools/tools.py"
            elif "rag" in task_lower or "retrieve" in task_lower:
                path = "rag/retrieve.py"
            elif state.retrieved_context:
                path = state.retrieved_context[0].get("file_path", "main.py")
            else:
                path = "main.py"

            return {
                "path": path,
                "start_line": None,
                "end_line": None,
            }

        if tool_name == "identify_error":
            error_log = "\n".join(state.errors_seen[-3:]) if state.errors_seen else ""
            return {
                "error_log": error_log,
            }

        if tool_name == "apply_patch":
            return {
                "patch": "",
            }

        if tool_name == "replace_in_file":
            return {
                "path": "main.py",
                "old_text": "",
                "new_text": "",
                "replace_all": False,
            }

        return {}

    def _execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        state: AgentState,
    ) -> ToolResult:
        if not tool_name:
            return ToolResult(
                tool_name="unknown",
                success=False,
                output="",
                error="Missing tool_name.",
            )

        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool not found: {tool_name}",
            )

        arguments = self._normalize_tool_arguments(tool_name, arguments or {})
        schema_error = self._validate_tool_arg(tool_name, arguments)
        if schema_error:
            print(f"    ✗ [SCHEMA] {tool_name}: {schema_error}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Schema validation failed: {schema_error}"
            )

        self._log_tool_call(tool_name, arguments)

        tool_fn = self.tools[tool_name]

        try:
            result = tool_fn(state=state, **arguments)

            if isinstance(result, ToolResult):
                self._log_tool_result(result)
                return result

            if isinstance(result, dict):
                r = ToolResult(
                    tool_name=tool_name,
                    success=bool(result.get("success", True)),
                    output=str(result.get("output", "")),
                    error=result.get("error"),
                    metadata=result.get("metadata", {}),
                )
                self._log_tool_result(r)
                return r

            r = ToolResult(tool_name=tool_name, success=True, output=str(result))
            self._log_tool_result(r)
            return r

        except Exception as exc:
            r = ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"{type(exc).__name__}: {str(exc)}",
                metadata={"tool_name": tool_name, "arguments": arguments},
            )
            self._log_tool_result(r)
            return r

    def _log_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        key_args = {k: v for k, v in arguments.items()
                    if v is not None and k not in {"include_hidden", "replace_all"}}
        args_str = ", ".join(
            f"{k}={repr(v)[:200] if isinstance(v, str) else repr(v)}"
            for k, v in list(key_args.items())[:3]
        )
        print(f"    → [{tool_name}] {args_str}")

    def _log_tool_result(self, result: ToolResult) -> None:
        if result.success:
            preview = (result.output or "").replace("\n", " ").strip()[:100]
            print(f"    ✓ {preview or '(no output)'}")
        else:
            err = (result.error or "unknown error")[:120]
            print(f"    ✗ {err}")

    def _normalize_tool_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Make LLM-generated tool arguments more tolerant.

        Examples:
        - list_files(path=".") -> list_files(directory=".")
        - read_file(file_path="main.py") -> read_file(path="main.py")
        - run_command(cmd="ls") -> run_command(command="ls")
        """
        args = dict(arguments or {})

        if tool_name == "list_files":
            if "path" in args and "directory" not in args:
                args["directory"] = args.pop("path")

            if "dir" in args and "directory" not in args:
                args["directory"] = args.pop("dir")

            allowed = {"directory", "max_depth", "include_hidden"}
            args = {k: v for k, v in args.items() if k in allowed}

            args.setdefault("directory", ".")
            args.setdefault("max_depth", 3)
            args.setdefault("include_hidden", False)

        elif tool_name == "read_file":
            if "file_path" in args and "path" not in args:
                args["path"] = args.pop("file_path")

            if "filename" in args and "path" not in args:
                args["path"] = args.pop("filename")

            allowed = {"path", "start_line", "end_line"}
            args = {k: v for k, v in args.items() if k in allowed}

            args.setdefault("path", "main.py")

        elif tool_name == "search_code":
            if "keyword" in args and "query" not in args:
                args["query"] = args.pop("keyword")

            if "pattern" in args and "query" not in args:
                args["query"] = args.pop("pattern")

            allowed = {"query", "include_glob", "max_results", "use_regex"}
            args = {k: v for k, v in args.items() if k in allowed}

            args.setdefault("query", "")
            args.setdefault("include_glob", "*.py")
            args.setdefault("max_results", 50)
            args.setdefault("use_regex", False)

        elif tool_name == "retrieve_context":
            if "question" in args and "query" not in args:
                args["query"] = args.pop("question")

            allowed = {"query", "top_k", "force_rebuild"}
            args = {k: v for k, v in args.items() if k in allowed}

            args.setdefault("query", "")
            args.setdefault("top_k", 6)
            args.setdefault("force_rebuild", False)

        elif tool_name == "run_command":
            if "cmd" in args and "command" not in args:
                args["command"] = args.pop("cmd")

            allowed = {"command", "timeout"}
            args = {k: v for k, v in args.items() if k in allowed}

            args.setdefault("command", "pwd")
            args.setdefault("timeout", 60)

        elif tool_name == "run_tests":
            allowed = {"test_path", "timeout"}
            args = {k: v for k, v in args.items() if k in allowed}

            args.setdefault("test_path", None)
            args.setdefault("timeout", 60)

        elif tool_name == "identify_error":
            if "error" in args and "error_log" not in args:
                args["error_log"] = args.pop("error")

            if "traceback" in args and "error_log" not in args:
                args["error_log"] = args.pop("traceback")

            allowed = {"error_log"}
            args = {k: v for k, v in args.items() if k in allowed}

            args.setdefault("error_log", "")

        elif tool_name == "apply_patch":
            allowed = {"patch"}
            args = {k: v for k, v in args.items() if k in allowed}

            args.setdefault("patch", "")

        elif tool_name == "replace_in_file":
            if "file_path" in args and "path" not in args:
                args["path"] = args.pop("file_path")

            allowed = {"path", "old_text", "new_text", "replace_all"}
            args = {k: v for k, v in args.items() if k in allowed}

            args.setdefault("path", "main.py")
            args.setdefault("old_text", "")
            args.setdefault("new_text", "")
            args.setdefault("replace_all", False)

        elif tool_name == "write_file":
            if "file_path" in args and "path" not in args:
                args["path"] = args.pop("file_path")

            allowed = {"path", "content", "overwrite"}
            args = {k: v for k, v in args.items() if k in allowed}

            args.setdefault("path", "output.txt")
            args.setdefault("content", "")
            args.setdefault("overwrite", False)

        return args

    def _build_messages(self, step: PlanStep, state: AgentState) -> List[Dict[str, str]]:
        system_prompt = self._system_prompt()

        relevant_skills = self.skill_registry.find_relevant(
            query=step.task,
            errors=state.errors_seen[-3:],
        )
        skills_text = ""
        if relevant_skills:
            skills_text = "Relevant skills:\n" + "\n\n".join(
                f"[{s.name}]\n{s.summary}" for s in relevant_skills
            )

        relevant_memory = self.memory.retrieve_relevant(query=step.task, top_k=3)
        memory_text = self.memory.summarize_short_term(max_items=3, max_chars=1500)
        if relevant_memory:
            memory_text += "\n\nRelevant past insights:\n" + "\n".join(
                f"-[{m.memory_type}]{m.content}" for m in relevant_memory
            )

        user_prompt = f"""
User request:
{state.input_query}

Current plan:
{state.plan_summary()}

Current step:
Step {step.step_id}: {step.task}

Expected output:
{step.expected_output}

Suggested tools:
{step.suggested_tools}

Retrieved context:
{state.retrieved_context_text(max_chunks=3, max_chars_per_chunk=800)}

Memory:
{memory_text}

Recent tool results:
{state.recent_tool_summary(limit=3)}

Available skills:
{skills_text}

Available tools:
{self._tool_descriptions()}

Instructions:
- Complete the current step only.
- If you need a tool, return only JSON.
- If you have enough information to finish this step, return the final answer directly.
- Do not invent file contents.
- Read files or retrieve context before making conclusions.
- For summarization steps, after reading/retrieving context, return a concise natural-language summary.
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _system_prompt(self) -> str:
        return """
You are the Executor of a coding agent.

Your job:
- Complete exactly one plan step at a time.
- Use tools to inspect, modify, and validate the repository.
- Be careful and grounded.
- Never claim code was changed unless a tool successfully changed it.
- Never mark validation as passed unless tests or syntax checks pass.

Important tool rules:
- For list_files, use arguments: {"directory": ".", "max_depth": 3}
- For read_file, use arguments: {"path": "relative/path.py"}
- For search_code, use arguments: {"query": "keyword", "include_glob": "*.py"}
- For retrieve_context, use arguments: {"query": "natural language query", "top_k": 5}
- For run_command, use arguments: {"command": "safe shell command", "timeout": 60}
- For project summary, architecture analysis, implementation status, or missing-parts analysis, prefer retrieve_context over search_code.
- Do not summarize only from file names. Read files or retrieve context before making conclusions.

When you need to use a tool, return ONLY a valid JSON object.
Do not wrap it in markdown.
Do not add explanation before or after the JSON.

Tool-call JSON schema:
{
  "tool_name": "read_file",
  "arguments": {
    "path": "core/executor.py"
  }
}

If the current step is fully complete and no more tool call is needed, return a concise final message.
""".strip()

    def _tool_descriptions(self) -> str:
        if not self.tools:
            return "No tools are currently registered."

        lines = []
        for name, fn in self.tools.items():
            doc = getattr(fn, "__doc__", "") or ""
            doc = doc.strip().replace("\n", " ")
            lines.append(f"- {name}: {doc[:220]}")
        return "\n".join(lines)

    def _normalize_llm_response(self, response: Any) -> str:
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

    def _parse_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON tool call from LLM output.

        Expected:
        {
          "tool_name": "read_file",
          "arguments": {"path": "main.py"}
        }
        """
        if not content:
            return None

        content = content.strip()

        # Remove common markdown wrappers.
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        try:
            data = json.loads(content)
            if isinstance(data, dict) and "tool_name" in data:
                if "arguments" not in data or data["arguments"] is None:
                    data["arguments"] = {}
                return data
        except json.JSONDecodeError:
            pass

        # Extract first JSON object from a mixed response.
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return None

        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "tool_name" in data:
                if "arguments" not in data or data["arguments"] is None:
                    data["arguments"] = {}
                return data
        except json.JSONDecodeError:
            return None

        return None

    def _tool_result_is_enough(
        self,
        step: PlanStep,
        result: ToolResult,
        used_tools: Optional[set[str]] = None,
    ) -> bool:
        """
        Decide whether one tool result is enough to complete this step.

        For analysis/summarization steps, the LLM should read/retrieve context
        before producing a final natural-language answer.
        """
        used_tools = used_tools or set()

        if not result.success:
            return False

        task = step.task.lower()

        is_summary_task = any(
            keyword in task
            for keyword in [
                "summarize",
                "summary",
                "explain",
                "analyze",
                "identify what is missing",
                "what is implemented",
                "bullet",
                "project",
            ]
        )

        if is_summary_task:
            if not used_tools.intersection({"retrieve_context", "read_file"}):
                return False

            if result.tool_name in {
                "read_file",
                "retrieve_context",
                "search_code",
                "list_files",
                "run_command",
            }:
                return False

        if result.tool_name in {"list_files", "search_code", "read_file", "retrieve_context"}:
            return True

        if result.tool_name in {"apply_patch", "replace_in_file", "write_file"}:
            return False

        if result.tool_name in {"run_tests", "run_command"}:
            return result.success

        return True

    def _build_final_answer(self, state: AgentState) -> str:
        """
        If an LLM client is available, ask it to write a concise summary.
        Otherwise produce a compact rule-based summary (no raw tool dumps).
        """
        if self.client:
            return self._llm_final_summary(state)
        return self._rule_based_summary(state)

    def _llm_final_summary(self, state: AgentState) -> str:
        """Ask the LLM to write a 3-5 line result summary."""
        # Collect compact evidence: what was read, modified, tested
        evidence_lines = [f"Task: {state.input_query}"]
        if state.files_read:
            evidence_lines.append("Files inspected: " + ", ".join(state.files_read))
        if state.files_modified:
            evidence_lines.append("Files modified: " + ", ".join(state.files_modified))

        # Key symbols from retrieved context (non-knowledge-base only)
        symbols = []
        for item in state.retrieved_context[:10]:
            sym = item.get("symbol_name")
            fp = item.get("file_path", "")
            if sym and not fp.startswith("__knowledge__"):
                symbols.append(f"{sym} ({fp})")
        if symbols:
            evidence_lines.append("Key symbols: " + ", ".join(dict.fromkeys(symbols)))

        # Last test outcome
        if state.test_results:
            evidence_lines.append("Last test output: " + state.test_results[-1][:300])

        # Step outcomes (just task + status, no raw output)
        step_lines = []
        for s in state.plan:
            step_lines.append(f"  Step {s.step_id} [{s.status.value}]: {s.task}")
            if s.status == StepStatus.FAILED and s.error:
                step_lines.append(f"    Error: {s.error[:120]}")
        evidence_lines.append("Steps:\n" + "\n".join(step_lines))

        evidence = "\n".join(evidence_lines)

        prompt = (
            "You are writing the final result card for an autonomous coding agent run.\n\n"
            "STRICT RULES — violating any rule makes your answer worthless:\n"
            "1. Only state facts that appear verbatim in the Evidence section below.\n"
            "2. Every claim must be traceable to a specific file name or line in the evidence.\n"
            "3. If the evidence does not contain enough information to answer, write exactly:\n"
            "   'Insufficient evidence — the agent did not read the files needed to answer this.'\n"
            "   Do NOT guess, infer, or fill gaps with plausible-sounding details.\n"
            "4. Do NOT mention tool names, step numbers, or internal agent mechanics.\n"
            "5. Max 6 lines. Plain English, no markdown headers.\n\n"
            "FORMAT:\n"
            "- Bug fix: one line per change (file, what changed, test result).\n"
            "- Project review: one line saying what the project does (cite the file you got this from),\n"
            "  then bullet the main components you actually read.\n\n"
            f"Evidence:\n{evidence}"
        )

        try:
            return self.client.chat([{"role": "user", "content": prompt}])
        except Exception:
            return self._rule_based_summary(state)

    def _rule_based_summary(self, state: AgentState) -> str:
        """Compact summary without raw tool output dumps."""
        completed = [s for s in state.plan if s.status == StepStatus.COMPLETED]
        failed = [s for s in state.plan if s.status == StepStatus.FAILED]
        lines = []

        if state.files_modified:
            lines.append(f"Modified {len(state.files_modified)} file(s): " + ", ".join(state.files_modified))
        else:
            lines.append("No files were modified.")

        if state.test_results:
            last = state.test_results[-1]
            snippet = last.replace("\n", " ").strip()[:200]
            lines.append(f"Tests: {snippet}")

        lines.append(f"Steps: {len(completed)} completed, {len(failed)} failed out of {len(state.plan)}.")

        if failed:
            lines.append("Failed steps: " + "; ".join(
                f"step {s.step_id} — {(s.error or 'unknown')[:80]}" for s in failed
            ))

        # Key symbols found (for review tasks)
        symbols = []
        for item in state.retrieved_context[:8]:
            sym = item.get("symbol_name")
            fp = item.get("file_path", "")
            if sym and not fp.startswith("__knowledge__"):
                entry = f"{sym} ({fp})"
                if entry not in symbols:
                    symbols.append(entry)
        if symbols and not state.files_modified:
            lines.append("Key symbols found: " + ", ".join(symbols[:6]))

        return "\n".join(lines)