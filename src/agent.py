import argparse
import json
from pathlib import Path

from llm import LLMClient, TOOLS
from tool import (
    list_files,
    search_code,
    read_file,
    write_file,
    run_command,
    finish,
    choose_validation_command,
)


SYSTEM_PROMPT = """
You are an autonomous debugging agent working on a local code repository.

Your goals:
1. Understand the repository and reproduce the issue.
2. Use tools to inspect files and run commands.
3. Identify the root cause with evidence.
4. Make the smallest correct fix.
5. Re-run validation after every edit.
6. Only finish when validation succeeds.

Rules:
- Do not claim success without re-running validation.
- Use tools before editing unless the failure is already obvious.
- Prefer minimal, surgical fixes.
- Do not modify unrelated files.
- If there are no tests, use the provided validation command, build, compile, or runtime reproduction.
- Keep changes local and practical.
"""


TOOL_MAP = {
    "list_files": list_files,
    "search_code": search_code,
    "read_file": read_file,
    "run_command": run_command,
    "write_file": write_file,
    "finish": finish,
}


def execute_tool_call(tool_name: str, arguments: dict):
    fn = TOOL_MAP.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        return fn(**arguments)
    except Exception as e:
        return {"error": repr(e)}


def _normalize_tool_args(tool_name: str, tool_args: dict, repo_root: str, timeout: int) -> dict:
    repo_path = Path(repo_root).resolve()

    if tool_name in {"read_file", "write_file"}:
        path = tool_args.get("path")
        if path and not Path(path).is_absolute():
            tool_args["path"] = str((repo_path / path).resolve())

    if tool_name == "run_command":
        if not tool_args.get("cwd"):
            tool_args["cwd"] = repo_root
        if "timeout" not in tool_args:
            tool_args["timeout"] = timeout

    if tool_name in {"list_files", "search_code"}:
        if not tool_args.get("root"):
            tool_args["root"] = repo_root

    return tool_args


def main():
    parser = argparse.ArgumentParser(description="Tool-calling local debugging agent")
    parser.add_argument("--repo", required=True, help="Path to target repository")
    parser.add_argument("--repro-cmd", default=None, help="Optional validation/reproduction command")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum number of agent steps")
    parser.add_argument("--timeout", type=int, default=90, help="Command timeout in seconds")
    args = parser.parse_args()

    repo_root = str(Path(args.repo).resolve())
    repo_path = Path(repo_root)

    if not repo_path.exists():
        raise RuntimeError(f"Repository not found: {repo_root}")

    validation_cmd = choose_validation_command(repo_root, args.repro_cmd)
    client = LLMClient()

    state = {
        "repo_root": repo_root,
        "validation_cmd": validation_cmd,
        "last_validation_passed": False,
        "last_validation_output": "",
        "edited_files": [],
    }

    initial_validation = run_command(
        cmd=validation_cmd,
        cwd=repo_root,
        timeout=args.timeout,
    )
    state["last_validation_output"] = json.dumps(initial_validation, indent=2, ensure_ascii=False)
    state["last_validation_passed"] = (initial_validation.get("returncode", -1) == 0)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"You are debugging this repository: {repo_root}\n"
                f"Preferred validation command: {validation_cmd}\n"
                "Use tools to inspect the repo, understand the failure, fix it, and validate again."
            ),
        },
        {
            "role": "user",
            "content": (
                "Initial validation result:\n"
                f"{json.dumps(initial_validation, indent=2, ensure_ascii=False)}\n\n"
                "Required workflow:\n"
                "1. Inspect evidence\n"
                "2. Read relevant files\n"
                "3. Edit code only when you have a likely root cause\n"
                "4. After every code edit, run the validation command again\n"
                "5. Only call finish after validation passes"
            ),
        },
    ]

    if state["last_validation_passed"]:
        print("[INFO] Initial validation already passed. No reproducible failure found yet.")
        print(json.dumps(initial_validation, indent=2, ensure_ascii=False))

    for step in range(1, args.max_steps + 1):
        print(f"\n========== STEP {step} ==========")

        response = client.chat(
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = response["choices"][0]["message"]
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            assistant_text = msg.get("content", "")
            print("[ASSISTANT]")
            print(assistant_text or "(empty)")
            messages.append({
                "role": "assistant",
                "content": assistant_text,
            })
            continue

        messages.append({
            "role": "assistant",
            "content": msg.get("content", ""),
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]

            try:
                tool_args = json.loads(raw_args)
            except json.JSONDecodeError:
                tool_args = {}
                result = {"error": f"Invalid JSON arguments for tool {tool_name}: {raw_args}"}
            else:
                tool_args = _normalize_tool_args(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    repo_root=repo_root,
                    timeout=args.timeout,
                )

                print(f"[TOOL CALL] {tool_name}({tool_args})")
                result = execute_tool_call(tool_name, tool_args)

            if tool_name == "run_command":
                state["last_validation_output"] = json.dumps(result, indent=2, ensure_ascii=False)
                state["last_validation_passed"] = (result.get("returncode", -1) == 0)

            if tool_name == "write_file":
                written_path = tool_args.get("path")
                if written_path:
                    try:
                        rel = str(Path(written_path).resolve().relative_to(repo_path))
                    except Exception:
                        rel = str(written_path)
                    state["edited_files"].append(rel)

            preview = json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, (dict, list)) else str(result)

            print("[TOOL RESULT]")
            print(preview[:4000])

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tool_name,
                "content": preview,
            })
            if tool_name == "finish":
                if state["last_validation_passed"]:
                    print("\n[SUCCESS]")
                    print(preview)
                    return
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            "You attempted to finish, but the latest validation has not passed yet. "
                            "Continue debugging.\n\n"
                            "Latest validation result:\n"
                            f"{state['last_validation_output']}"
                        ),
                    })

    print("\n[STOP] Reached max steps.")
    print("Last validation passed:", state["last_validation_passed"])
    print("Edited files:", state["edited_files"])
    print("Validation command:", state["validation_cmd"])
    print("Last validation output:")
    print(state["last_validation_output"][:4000])


if __name__ == "__main__":
    main()