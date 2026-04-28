from __future__ import annotations

import argparse

from core.state import AgentState
from core.planner import Planner
from core.executor import Executor
from core.memory import AgentMemory
from tools.tools import get_tool_map
from llm import create_local_llm_client


def check_validation(state: AgentState) -> bool:
    if not state.test_results:
        return True

    latest_test = state.test_results[-1]
    return latest_test.success


def run_agent(task_description: str, repo_root: str, client=None):
    state = AgentState(
        input_query=task_description,
        repo_root=repo_root,
        max_steps=20,
    )

    memory = AgentMemory()
    tools = get_tool_map()

    planner = Planner(client=client)
    executor = Executor(
        client=client,
        tools=tools,
        memory=memory,
    )

    planner.create_initial_plan(state)

    loop_count = 0
    max_loops = 20

    while loop_count < max_loops:
        loop_count += 1

        print("\n==============================")
        print("[*] Current Plan")
        print(state.plan_summary())

        current_step = state.get_current_step()

        if current_step is None:
            state.is_finished = True
            state.final_answer = executor._build_final_answer(state)
            break

        state = executor.execute_current_step(state)

        if state.test_results and not check_validation(state):
            print("[!] Validation failed. Re-planning...")
            planner.adjust_plan(state)

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