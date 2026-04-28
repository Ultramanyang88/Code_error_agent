from __future__ import annotations

from typing import Any, Dict, List


TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "list_files": {
        "description": "List files under the repository root.",
        "when_to_use": [
            "Use this at the beginning to understand repository structure.",
            "Use this when you do not know which file to inspect.",
            "Use this before search/read if the project structure is unclear.",
        ],
        "when_not_to_use": [
            "Do not use this to inspect file content.",
            "Do not use this repeatedly if the repo structure is already known.",
        ],
        "parameters": {
            "directory": {
                "type": "string",
                "description": "Relative directory path from repo root.",
                "default": ".",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum recursive depth.",
                "default": 3,
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Whether to include hidden files and directories.",
                "default": False,
            },
        },
        "output": "A tree-like listing of files and directories.",
    },

    "read_file": {
        "description": "Read a file from the repository.",
        "when_to_use": [
            "Use this when you know the exact file path.",
            "Use this after search_code or retrieve_context identifies a relevant file.",
            "Use line ranges for large files.",
        ],
        "when_not_to_use": [
            "Do not use this for binary files.",
            "Do not read very large files without line ranges.",
            "Do not use this to search across the whole repo; use search_code instead.",
        ],
        "parameters": {
            "path": {
                "type": "string",
                "description": "Relative file path from repo root.",
                "required": True,
            },
            "start_line": {
                "type": "integer",
                "description": "Optional 1-based start line.",
                "default": None,
            },
            "end_line": {
                "type": "integer",
                "description": "Optional 1-based end line.",
                "default": None,
            },
        },
        "output": "File content with line numbers.",
    },

    "search_code": {
        "description": "Search repository files using keyword or regex.",
        "when_to_use": [
            "Use this to find class names, function names, imports, config keys, or error messages.",
            "Use this before reading files if the relevant file is unknown.",
            "Use this for exact symbol search.",
        ],
        "when_not_to_use": [
            "Do not use this for broad semantic questions; use retrieve_context later.",
            "Do not use extremely generic queries like 'the' or 'class'.",
        ],
        "parameters": {
            "query": {
                "type": "string",
                "description": "Keyword or regex pattern.",
                "required": True,
            },
            "include_glob": {
                "type": "string",
                "description": "Optional file glob, e.g. '*.py'.",
                "default": "*.py",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matches.",
                "default": 50,
            },
            "use_regex": {
                "type": "boolean",
                "description": "Whether query should be treated as regex.",
                "default": False,
            },
        },
        "output": "Matched file paths, line numbers, and line content.",
    },

    "write_file": {
        "description": "Write full content to a repository file.",
        "when_to_use": [
            "Use this to create a new file.",
            "Use this when replacing a small generated file completely.",
        ],
        "when_not_to_use": [
            "Avoid using this to rewrite large existing files.",
            "For small modifications, prefer replace_in_file or apply_patch.",
        ],
        "parameters": {
            "path": {
                "type": "string",
                "description": "Relative file path from repo root.",
                "required": True,
            },
            "content": {
                "type": "string",
                "description": "Full file content to write.",
                "required": True,
            },
            "overwrite": {
                "type": "boolean",
                "description": "Whether to overwrite if file exists.",
                "default": False,
            },
        },
        "output": "Write status and changed file path.",
    },

    "replace_in_file": {
        "description": "Replace exact text inside a file.",
        "when_to_use": [
            "Use this for small focused edits.",
            "Use this when you know the exact old_text to replace.",
        ],
        "when_not_to_use": [
            "Do not use if old_text is ambiguous.",
            "Do not use if the file has not been read or searched first.",
        ],
        "parameters": {
            "path": {
                "type": "string",
                "description": "Relative file path from repo root.",
                "required": True,
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to replace.",
                "required": True,
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text.",
                "required": True,
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences or only first occurrence.",
                "default": False,
            },
        },
        "output": "Replacement status and number of replacements.",
    },

    "apply_patch": {
        "description": "Apply a unified diff patch to repository files.",
        "when_to_use": [
            "Use this for multi-line code edits.",
            "Use this when modifying several related blocks.",
            "Use this after reading the target file.",
        ],
        "when_not_to_use": [
            "Do not use this before inspecting the target files.",
            "Do not use for unrelated large rewrites.",
        ],
        "parameters": {
            "patch": {
                "type": "string",
                "description": "Unified diff patch.",
                "required": True,
            },
        },
        "output": "Patch application result and changed files.",
    },

    "run_command": {
        "description": "Run a safe shell command inside the repository.",
        "when_to_use": [
            "Use this for syntax checks, tests, linting, or dependency inspection.",
            "Use this when the exact command is known.",
        ],
        "when_not_to_use": [
            "Do not run destructive commands.",
            "Do not run network installation commands unless explicitly allowed.",
            "Do not run commands outside the repository.",
        ],
        "parameters": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
                "required": True,
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds.",
                "default": 60,
            },
        },
        "output": "exit_code, stdout, stderr.",
    },

    "run_tests": {
        "description": "Run tests or a basic Python syntax validation.",
        "when_to_use": [
            "Use after code changes.",
            "Use when validating whether the repository still works.",
            "Use when debugging failing tests.",
        ],
        "when_not_to_use": [
            "Do not mark a step completed if tests fail.",
        ],
        "parameters": {
            "test_path": {
                "type": "string",
                "description": "Optional test file or directory.",
                "default": None,
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds.",
                "default": 60,
            },
        },
        "output": "Test result, stdout, stderr, exit_code.",
    },

    "identify_error": {
        "description": "Analyze traceback or command output and identify likely root cause.",
        "when_to_use": [
            "Use after run_tests or run_command fails.",
            "Use before editing code after an error.",
        ],
        "when_not_to_use": [
            "Do not use this if there is no error log.",
        ],
        "parameters": {
            "error_log": {
                "type": "string",
                "description": "Raw traceback, stderr, or failed test output.",
                "required": True,
            },
        },
        "output": "Structured error analysis.",
    },

    "git_diff": {
        "description": "Show current git diff.",
        "when_to_use": [
            "Use after edits to summarize changes.",
            "Use before final answer.",
        ],
        "when_not_to_use": [
            "Do not use this before any edits.",
        ],
        "parameters": {},
        "output": "Current git diff.",
    },

    "retrieve_context": {
        "description": "Retrieve relevant repository context using FAISS-based hybrid RAG.",
        "when_to_use": [
            "Use this when the relevant file is unknown.",
            "Use this for broad semantic questions about the codebase.",
            "Use this before editing code when you need repository-level context.",
            "Use this for debugging, architecture analysis, and locating implementation gaps.",
        ],
        "when_not_to_use": [
            "Do not use this for exact known file reads; use read_file.",
            "Do not use this for exact symbol matching only; use search_code first.",
        ],
        "parameters": {
            "query": {
                "type": "string",
                "description": "Natural language retrieval query.",
                "required": True,
            },
            "top_k": {
                "type": "integer",
                "description": "Number of reranked chunks to return.",
                "default": 6,
            },
            "force_rebuild": {
                "type": "boolean",
                "description": "Whether to rebuild the FAISS index before retrieval.",
                "default": False,
            },
        },
        "output": "Relevant code chunks with file path, line range, symbol name, and score.",
    },
}


def get_tool_specs() -> Dict[str, Dict[str, Any]]:
    return TOOL_SPECS


def format_tool_specs_for_prompt(tool_names: List[str] | None = None) -> str:
    """
    Convert tool specs into a readable manual for the LLM executor prompt.
    """
    selected = TOOL_SPECS

    if tool_names:
        selected = {
            name: spec
            for name, spec in TOOL_SPECS.items()
            if name in tool_names
        }

    blocks = []

    for name, spec in selected.items():
        params = spec.get("parameters", {})
        when_to_use = spec.get("when_to_use", [])
        when_not_to_use = spec.get("when_not_to_use", [])

        block = [
            f"Tool: {name}",
            f"Description: {spec.get('description', '')}",
            "When to use:",
        ]

        for item in when_to_use:
            block.append(f"- {item}")

        if when_not_to_use:
            block.append("When not to use:")
            for item in when_not_to_use:
                block.append(f"- {item}")

        block.append("Parameters:")
        if params:
            for param_name, param_spec in params.items():
                required = param_spec.get("required", False)
                default = param_spec.get("default", None)
                desc = param_spec.get("description", "")
                type_name = param_spec.get("type", "unknown")

                req_text = "required" if required else f"default={default}"
                block.append(
                    f"- {param_name}: {type_name}, {req_text}. {desc}"
                )
        else:
            block.append("- No parameters.")

        block.append(f"Output: {spec.get('output', '')}")

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)