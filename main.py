from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from the current/parent dirs into os.environ, if present -- no-op otherwise
except ImportError:
    pass

from core.state import AgentState, RunStatus, ValidationStatus
from core.planner import Planner
from core.executor import Executor
from core.memory import AgentMemory
from core.logging_setup import log_event
from tools.tools import get_tool_map
from llm import create_local_llm_client


def get_validation_status(state: AgentState) -> ValidationStatus:
    if not state.test_results:
        return ValidationStatus.NOT_RUN

    latest = state.test_results[-1]

    if latest.metadata.get("timed_out"):
        return ValidationStatus.ERROR

    if latest.metadata.get("execution_error"):
        return ValidationStatus.ERROR

    if latest.success:
        return ValidationStatus.PASSED

    return ValidationStatus.FAILED


MAX_TASK_CHARS = 6000  # keep pasted-in text bounded; it gets echoed into every single prompt


def _bound_task_description(task_description: str) -> str:
    if len(task_description) <= MAX_TASK_CHARS:
        return task_description
    dropped = len(task_description) - MAX_TASK_CHARS
    return (
        task_description[:MAX_TASK_CHARS]
        + f"\n...[truncated {dropped} more characters]\n"
        "Note: this input was too long to paste directly into the prompt. "
        "If you need to analyze the full document, save it as a file in the repo "
        "and use read_file/retrieve_context instead of relying on this pasted text."
    )


def _load_mcp_tools(repo_root: str) -> tuple:
    """
    Connect to every enabled server in the MCP config (mcp_servers.json /
    $MCP_SERVERS_CONFIG, falling back to the built-in filesystem server --
    see agent_mcp/config.py) and return (tools, configs): merged,
    allow/deny-filtered tool maps namespaced per server, plus the config
    list so the caller can build a ToolRegistry. Each server gets its own
    get_shared_mcp_client() connection, so one being unreachable doesn't
    take the others down with it.
    """
    from agent_mcp.client import get_shared_mcp_client
    from agent_mcp.config import load_mcp_server_configs, filter_tool_map

    tools: dict = {}

    try:
        configs = load_mcp_server_configs()
    except Exception as e:
        print(f"[!] Failed to load MCP server config: {e}")
        return tools, []

    for cfg in configs:
        if not cfg.enabled:
            continue
        try:
            # cwd_from_repo_root: some servers (e.g. the filesystem server)
            # resolve relative args like "." against the subprocess's own
            # working directory, so that must be the repo being worked on --
            # not wherever this process happened to be launched from.
            client = get_shared_mcp_client(
                command=cfg.command,
                args=cfg.args,
                namespace=cfg.namespace,
                cwd=repo_root if cfg.cwd_from_repo_root else None,
            )
            server_tools = filter_tool_map(client.get_tool_map(), cfg)
            tools.update(server_tools)
        except Exception as e:
            print(f"[!] MCP server '{cfg.name}' unavailable: {e}")

    return tools, configs


def run_agent(
    task_description: str,
    repo_root: str,
    client=None,
    trace_path: Optional[str] = None,
    run_id: Optional[str] = None,
    step_callback=None,  # callable(event_type: str, data: dict)
    session_id: Optional[str] = None,
) -> AgentState:
    run_id = run_id or str(uuid.uuid4())[:8]
    started_at = time.time()

    state = AgentState(
        input_query=_bound_task_description(task_description),
        repo_root=repo_root,
    )
    # AgentState.run_id defaults to its own fresh uuid4 — without this, tool-call
    # log events (keyed on state.run_id) would use a different id than the
    # run_id this function returns / the API exposes at /api/run/{run_id},
    # making it impossible to correlate structured logs back to a specific run.
    state.run_id = run_id
    state.started_at = started_at

    tools = get_tool_map()
    namespace = AgentMemory.namespace_for(repo_root, session_id)
    memory = AgentMemory(persist_dir=str(Path(".agent_memory")/namespace))
    # Lets tools/tools.py's retrieve_context() include this session's memory
    # in its recall pool (rag/federated.py's FederatedRetriever) without
    # threading a `memory` param through every tool call.
    state.metadata["agent_memory"] = memory

    mcp_tools, mcp_configs = _load_mcp_tools(state.repo_root)
    tools.update(mcp_tools)

    from tools.registry import ToolRegistry
    tool_registry = ToolRegistry.build(tools, mcp_configs)

    planner = Planner(client=client, tool_registry=tool_registry)
    executor = Executor(
        client=client,
        tools=tools,
        memory=memory,
        tool_registry=tool_registry,
    )

    log_event(
        "run_started",
        run_id=run_id,
        session_id=session_id,
        repo_root=state.repo_root,
        task=state.input_query[:200],
    )

    planner.create_initial_plan(state)

    if step_callback:
        step_callback("plan_created", {
            "steps": [{"step_id": s.step_id, "task": s.task} for s in state.plan]
        })

    trace: list = []
    loop_count = 0
    max_loops = 20

    while loop_count < max_loops:
        loop_count += 1

        print("\n==============================")
        print("[*] Current Plan")
        print(state.plan_summary())

        current_step = state.get_current_step()

        if current_step is None:
            if state.has_unresolved_failures():
                state.run_status = RunStatus.FAILED
                state.stop_reason = "step_retry_exhausted"
            else:
                state.run_status = RunStatus.COMPLETED
            state.final_answer = executor._build_final_answer(state)
            break

        elapsed = time.time() - started_at
        if elapsed >= state.budget.deadline_seconds:
            state.run_status = RunStatus.FAILED
            state.stop_reason = "deadline_exceeded"
            state.final_answer = executor._build_final_answer(state)
            if step_callback:
                step_callback("abandoned", {"reason": "deadline_exceeded", "elapsed_s": round(elapsed, 1)})
            break
        
        if len(state.tool_history) >= state.budget.max_tool_calls:
            state.run_status = RunStatus.FAILED
            state.stop_reason = "tool_call_limit_exceeded"
            state.final_answer = executor._build_final_answer(state)
            if step_callback:
                step_callback("abandoned", {"reason": "tool_call_limit_exceeded"})
            break

        if step_callback:
            step_callback("step_start", {
                "step_id": current_step.step_id,
                "task": current_step.task,
            })

        state = executor.execute_current_step(state)

        # Record trajectory entry after each step
        executed = state.plan[-1] if state.plan else None
        for s in reversed(state.plan):
            if s.status.value in ("completed", "failed"):
                executed = s
                break
        if executed:
            entry = {
                "run_id": run_id,
                "loop": loop_count,
                "step_id": executed.step_id,
                "task": executed.task,
                "status": executed.status.value,
                "retry_count": executed.retry_count,
                "tool_results": [
                    {"tool": r.tool_name, "success": r.success, "error": r.error}
                    for r in executed.tool_results
                ],
                "error": executed.error,
                "timestamp": time.time(),
                "elapsed_s": round(time.time() - started_at, 2),
            }
            trace.append(entry)
            if step_callback:
                step_callback("step_done", entry)

        if state.test_results and get_validation_status(state) != ValidationStatus.PASSED:
            if state.replan_count >= state.budget.max_replans:
                print("[!] Replan limit reached. Abandoning.")
                state.run_status = RunStatus.FAILED
                state.stop_reason = "replan_limit_exceeded"
                state.final_answer = executor._build_final_answer(state)
                if step_callback:
                    step_callback("abandoned", {"reason": "replan_limit_exceeded"})
                break
            print("[!] Validation failed. Replanning...")
            planner.adjust_plan(state)
            if step_callback:
                step_callback("replan", {
                    "replan_count": state.replan_count,
                    "steps": [{"step_id": s.step_id, "task": s.task} for s in state.plan],
                })

    state.finished_at = time.time()

    log_event(
        "run_finished",
        run_id=run_id,
        session_id=session_id,
        run_status=state.run_status.value,
        stop_reason=state.stop_reason,
        validation=state.validation_status.value,
        replan_count=state.replan_count,
        tool_call_count=len(state.tool_history),
        files_modified=len(state.files_modified),
        elapsed_ms=round((state.finished_at - started_at) * 1000, 2),
    )

    if step_callback:
        # Field set must match api/server.py's _runs[run_id]["result"] shape
        # -- they used to differ, so the SSE "done" event and the polled
        # /api/run/{id} result disagreed on what fields existed, and
        # fillAgentBubble() in index.html showed "undefined / undefined
        # steps" for live-streamed runs.
        step_callback("done", {
            "validation": state.validation_status.value,
            "run_status": state.run_status.value,
            "stop_reason": state.stop_reason,
            "replan_count": state.replan_count,
            "tool_call_count": len(state.tool_history),
            "steps_completed": sum(1 for s in state.plan if s.status.value == "completed"),
            "steps_total": len(state.plan),
            "files_modified": state.files_modified,
            "files_read": state.files_read,
            "elapsed_s": round(state.finished_at - started_at, 2),
            "final_answer": state.final_answer or "",
        })

    # Write trajectory log
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as f:
            for entry in trace:
                f.write(json.dumps(entry) + "\n")

    print("\n==============================")
    print("[*] Final Answer")
    print(state.final_answer or executor._build_final_answer(state))

    return state


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use local LLM client instead of fallback rule-based mode.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="openai_compatible",
        choices=["openai_compatible", "ollama"],
        help="LLM provider type.",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="LLM server base URL.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder:7b",
        help="Model name.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=(
            "Analyze this coding agent repository. "
            "Identify the current project structure, inspect the executor, planner, tools, and RAG modules, "
            "then summarize what is implemented and what is still missing."
        ),
        help="Task for the coding agent.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    client = None

    if args.llm:
        client = create_local_llm_client(
            provider=args.provider,
            base_url=args.base_url,
            model=args.model,
        )

        print("[*] LLM mode enabled")
        print(f"[*] Provider: {args.provider}")
        print(f"[*] Base URL: {client.base_url}")
        print(f"[*] Model: {client.model}")

    else:
        print("[*] Fallback mode enabled. No LLM client is used.")

    run_agent(
        task_description=args.task,
        repo_root=".",
        client=client,
    )