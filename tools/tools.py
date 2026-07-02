from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import fnmatch
import os
import re
import subprocess
import sys
import tempfile
import textwrap

from collections import OrderedDict
from core.state import AgentState, ToolResult

# tool list:list_files, read_file, search_code, retrieve_context, write_file, replace_in_file, apply_patch, run_command, run_tests, identify_error, git_diff

_RAG_CACHE_MAX = 4
_rag_engine_cache: "OrderedDict[str, Any]" = OrderedDict()

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp", ".tiff",
    ".pdf", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp4", ".mp3", ".wav", ".mov", ".avi",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".pyc", ".pyo", ".pyd",
    ".class", ".o", ".a",
}

TEXT_FILE_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".hpp",
    ".c",
    ".h",
}

DANGEROUS_COMMAND_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{\s*:\|:",
    r">\s*/dev/sd[a-z]",
    r"\bchmod\s+-R\s+777\b",
    r"\bchown\s+-R\b",
    r"\bcurl\b.*\|\s*(bash|sh)",
    r"\bwget\b.*\|\s*(bash|sh)",
]


def _safe_path(state: AgentState, relative_path: str) -> Path:
    """
    Resolve a path safely inside repo root.
    """
    return state.repo_path(relative_path)


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _should_ignore(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def _is_probably_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_FILE_EXTENSIONS:
        return True

    if path.name in {
        "Dockerfile",
        "Makefile",
        "requirements.txt",
        ".gitignore",
        ".env.example",
        "README",
        "CHANGELOG",
        "CHANGES",
        "HISTORY",
        "LICENSE",
        "LICENCE",
        "CONTRIBUTING",
        "AUTHORS",
        "NOTICE",
        "INSTALL",
        "Pipfile",
        "Procfile",
    }:
        return True

    return False


def _get_rag_engine(repo_root: str, force_rebuild: bool = False):
    from rag.retrieve import RAGEngine

    key = str(Path(repo_root).resolve())

    if force_rebuild:
        _rag_engine_cache.pop(key, None)
    
    if key not in _rag_engine_cache:
        if len(_rag_engine_cache) >= _RAG_CACHE_MAX:
            _rag_engine_cache.popitem(last=False)
        _rag_engine_cache[key] = RAGEngine(
            repo_root=repo_root,
            index_dir=".agent_index",
            auto_load=True,
        )
    else:
        _rag_engine_cache.move_to_end(key)
    return _rag_engine_cache[key]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def list_files(
    state: AgentState,
    directory: str = ".",
    max_depth: int = 3,
    include_hidden: bool = False,
) -> ToolResult:
    """
    List files under a repository directory.
    """
    try:
        base = _safe_path(state, directory)

        if not base.exists():
            return ToolResult(
                tool_name="list_files",
                success=False,
                output="",
                error=f"Directory does not exist: {directory}",
            )

        if not base.is_dir():
            return ToolResult(
                tool_name="list_files",
                success=False,
                output="",
                error=f"Path is not a directory: {directory}",
            )

        root = Path(state.repo_root).resolve()
        lines: List[str] = []

        def walk(current: Path, depth: int) -> None:
            if depth > max_depth:
                return

            try:
                children = sorted(
                    current.iterdir(),
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                )
            except PermissionError:
                return

            for child in children:
                rel = child.relative_to(root)

                if _should_ignore(rel):
                    continue

                if not include_hidden and _is_hidden(rel):
                    continue

                if not child.is_dir() and child.suffix.lower() in BINARY_EXTENSIONS:
                    continue

                indent = "  " * depth
                suffix = "/" if child.is_dir() else ""
                lines.append(f"{indent}{rel}{suffix}")

                if child.is_dir():
                    walk(child, depth + 1)

        walk(base, 0)

        output = "\n".join(lines) if lines else "(empty directory)"

        return ToolResult(
            tool_name="list_files",
            success=True,
            output=output,
            metadata={
                "directory": directory,
                "max_depth": max_depth,
            },
        )

    except Exception as exc:
        return ToolResult(
            tool_name="list_files",
            success=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
        )


def read_file(
    state: AgentState,
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> ToolResult:
    """
    Read a repository file with optional line range.
    """
    try:
        target = _safe_path(state, path)

        if not target.exists():
            return ToolResult(
                tool_name="read_file",
                success=False,
                output="",
                error=f"File does not exist: {path}",
                metadata={"path": path},
            )

        if not target.is_file():
            return ToolResult(
                tool_name="read_file",
                success=False,
                output="",
                error=f"Path is not a file: {path}",
                metadata={"path": path},
            )

        if not _is_probably_text_file(target):
            return ToolResult(
                tool_name="read_file",
                success=False,
                output="",
                error=f"File does not look like a supported text/code file: {path}",
                metadata={"path": path},
            )

        content = _read_text_file(target)
        lines = content.splitlines()

        total_lines = len(lines)

        if start_line is None:
            start_line = 1

        if end_line is None:
            end_line = total_lines

        start_line = max(1, int(start_line))
        end_line = min(total_lines, int(end_line))

        if start_line > end_line:
            return ToolResult(
                tool_name="read_file",
                success=False,
                output="",
                error=f"Invalid line range: {start_line}-{end_line}",
                metadata={"path": path},
            )

        selected = lines[start_line - 1:end_line]

        numbered = [
            f"{line_no:>4}: {line}"
            for line_no, line in enumerate(selected, start=start_line)
        ]

        output = "\n".join(numbered)

        return ToolResult(
            tool_name="read_file",
            success=True,
            output=output,
            metadata={
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": total_lines,
            },
        )

    except Exception as exc:
        return ToolResult(
            tool_name="read_file",
            success=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            metadata={"path": path},
        )


def search_code(
    state: AgentState,
    query: str,
    include_glob: str = "*.py",
    max_results: int = 50,
    use_regex: bool = False,
) -> ToolResult:
    """
    Search repository files using keyword or regex.
    """
    try:
        root = Path(state.repo_root).resolve()

        if not query or not query.strip():
            return ToolResult(
                tool_name="search_code",
                success=False,
                output="",
                error="Empty query.",
            )

        matches: List[str] = []
        pattern = re.compile(query) if use_regex else None

        for file_path in root.rglob("*"):
            if len(matches) >= max_results:
                break

            if not file_path.is_file():
                continue

            rel = file_path.relative_to(root)

            if _should_ignore(rel):
                continue

            if _is_hidden(rel):
                continue

            if include_glob and not fnmatch.fnmatch(str(rel), include_glob):
                continue

            if not _is_probably_text_file(file_path):
                continue

            try:
                lines = _read_text_file(file_path).splitlines()
            except Exception:
                continue

            for line_no, line in enumerate(lines, start=1):
                if use_regex:
                    found = bool(pattern.search(line))
                else:
                    found = query.lower() in line.lower()

                if found:
                    matches.append(f"{rel}:{line_no}: {line.strip()}")

                    if len(matches) >= max_results:
                        break

        output = "\n".join(matches) if matches else "No matches found."

        return ToolResult(
            tool_name="search_code",
            success=True,
            output=output,
            metadata={
                "query": query,
                "include_glob": include_glob,
                "max_results": max_results,
                "num_matches": len(matches),
                "use_regex": use_regex,
            },
        )

    except re.error as exc:
        return ToolResult(
            tool_name="search_code",
            success=False,
            output="",
            error=f"Invalid regex: {exc}",
            metadata={"query": query},
        )

    except Exception as exc:
        return ToolResult(
            tool_name="search_code",
            success=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            metadata={"query": query},
        )


def write_file(
    state: AgentState,
    path: str,
    content: str,
    overwrite: bool = False,
) -> ToolResult:
    """
    Write full content to a file.
    """
    try:
        target = _safe_path(state, path)

        if target.exists() and not overwrite:
            return ToolResult(
                tool_name="write_file",
                success=False,
                output="",
                error=f"File already exists and overwrite=False: {path}",
                metadata={"path": path},
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        return ToolResult(
            tool_name="write_file",
            success=True,
            output=f"Wrote file: {path}",
            metadata={
                "path": path,
                "changed_files": [path],
                "num_chars": len(content),
            },
        )

    except Exception as exc:
        return ToolResult(
            tool_name="write_file",
            success=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            metadata={"path": path},
        )


def replace_in_file(
    state: AgentState,
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> ToolResult:
    """
    Replace exact text inside a file.
    """
    try:
        target = _safe_path(state, path)

        if not target.exists() or not target.is_file():
            return ToolResult(
                tool_name="replace_in_file",
                success=False,
                output="",
                error=f"File does not exist: {path}",
                metadata={"path": path},
            )

        content = _read_text_file(target)

        if old_text not in content:
            return ToolResult(
                tool_name="replace_in_file",
                success=False,
                output="",
                error="old_text not found in file.",
                metadata={"path": path},
            )

        count = content.count(old_text)

        if replace_all:
            new_content = content.replace(old_text, new_text)
            replaced = count
        else:
            new_content = content.replace(old_text, new_text, 1)
            replaced = 1

        target.write_text(new_content, encoding="utf-8")

        return ToolResult(
            tool_name="replace_in_file",
            success=True,
            output=f"Updated {path}. Replacements made: {replaced}",
            metadata={
                "path": path,
                "changed_files": [path],
                "replacements": replaced,
                "total_occurrences": count,
            },
        )

    except Exception as exc:
        return ToolResult(
            tool_name="replace_in_file",
            success=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            metadata={"path": path},
        )


def apply_patch(
    state: AgentState,
    patch: str,
) -> ToolResult:
    """
    Apply a unified diff patch using git apply.
    """
    try:
        if not patch or not patch.strip():
            return ToolResult(
                tool_name="apply_patch",
                success=False,
                output="",
                error="Empty patch.",
            )

        root = Path(state.repo_root).resolve()

        changed_files = _extract_changed_files_from_patch(patch)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".patch",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(patch)
            patch_path = tmp.name

        try:
            check_cmd = ["git", "apply", "--check", patch_path]
            check_proc = subprocess.run(
                check_cmd,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if check_proc.returncode != 0:
                return ToolResult(
                    tool_name="apply_patch",
                    success=False,
                    output=check_proc.stdout,
                    error=check_proc.stderr or "git apply --check failed.",
                    metadata={"changed_files": changed_files},
                )

            apply_cmd = ["git", "apply", patch_path]
            apply_proc = subprocess.run(
                apply_cmd,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if apply_proc.returncode != 0:
                return ToolResult(
                    tool_name="apply_patch",
                    success=False,
                    output=apply_proc.stdout,
                    error=apply_proc.stderr or "git apply failed.",
                    metadata={"changed_files": changed_files},
                )

            return ToolResult(
                tool_name="apply_patch",
                success=True,
                output="Patch applied successfully.",
                metadata={"changed_files": changed_files},
            )

        finally:
            try:
                os.remove(patch_path)
            except OSError:
                pass

    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="apply_patch",
            success=False,
            output="",
            error="Patch command timed out.",
        )

    except Exception as exc:
        return ToolResult(
            tool_name="apply_patch",
            success=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
        )


def run_command(
    state: AgentState,
    command: str,
    timeout: int = 60,
) -> ToolResult:
    """
    Run a safe shell command inside the repository.
    """
    try:
        safety_error = _validate_command_safety(command)

        if safety_error:
            return ToolResult(
                tool_name="run_command",
                success=False,
                output="",
                error=safety_error,
                metadata={"command": command},
            )

        root = Path(state.repo_root).resolve()

        proc = subprocess.run(
            command,
            cwd=root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = _format_command_output(proc.stdout, proc.stderr, proc.returncode)

        return ToolResult(
            tool_name="run_command",
            success=proc.returncode == 0,
            output=output,
            error=None if proc.returncode == 0 else proc.stderr,
            metadata={
                "command": command,
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
        )

    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="run_command",
            success=False,
            output="",
            error=f"Command timed out after {timeout} seconds.",
            metadata={"command": command, "timeout": timeout},
        )

    except Exception as exc:
        return ToolResult(
            tool_name="run_command",
            success=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            metadata={"command": command},
        )


def run_tests(
    state: AgentState,
    test_path: Optional[str] = None,
    timeout: int = 60,
) -> ToolResult:
    """
    Run pytest if tests exist; otherwise run Python syntax compilation.
    """
    root = Path(state.repo_root).resolve()

    if test_path:
        command = f"{sys.executable} -m pytest {test_path}"
        result = run_command(state=state, command=command, timeout=timeout)
        result.tool_name = "run_tests"
        return result

    tests_dir = root / "tests"
    testcase_dir = root / "testcase"

    if tests_dir.exists():
        command = f"{sys.executable} -m pytest tests"
    elif testcase_dir.exists():
        command = f"{sys.executable} -m pytest testcase"
    else:
        command = f"{sys.executable} -m compileall ."

    result = run_command(state=state, command=command, timeout=timeout)
    result.tool_name = "run_tests"
    return result


def identify_error(
    state: AgentState,
    error_log: str,
) -> ToolResult:
    """
    Heuristically analyze an error log.
    """
    try:
        if not error_log or not error_log.strip():
            return ToolResult(
                tool_name="identify_error",
                success=False,
                output="",
                error="Empty error_log.",
            )

        analysis = {
            "error_type": "Unknown",
            "likely_root_cause": "Could not determine from heuristics.",
            "relevant_files": [],
            "suggested_next_steps": [],
        }

        text = error_log

        if "ModuleNotFoundError" in text:
            analysis["error_type"] = "ModuleNotFoundError"
            analysis["likely_root_cause"] = (
                "A module import path is wrong, missing, or not installed."
            )
            analysis["suggested_next_steps"] = [
                "Search for the missing module name.",
                "Check package/file names.",
                "Check relative imports and __init__.py files.",
            ]

        elif "ImportError" in text:
            analysis["error_type"] = "ImportError"
            analysis["likely_root_cause"] = (
                "A symbol cannot be imported from the target module."
            )
            analysis["suggested_next_steps"] = [
                "Search for the imported symbol.",
                "Check whether the function/class is defined.",
                "Check filename mismatch such as tool.py vs tools.py.",
            ]

        elif "NameError" in text:
            analysis["error_type"] = "NameError"
            analysis["likely_root_cause"] = (
                "A variable, function, or class is referenced before definition."
            )
            analysis["suggested_next_steps"] = [
                "Search for the missing name.",
                "Check scope and imports.",
            ]

        elif "AttributeError" in text:
            analysis["error_type"] = "AttributeError"
            analysis["likely_root_cause"] = (
                "An object does not have the referenced method or attribute."
            )
            analysis["suggested_next_steps"] = [
                "Check class definition.",
                "Check method name mismatch.",
                "Check whether the object type is correct.",
            ]

        elif "TypeError" in text:
            analysis["error_type"] = "TypeError"
            analysis["likely_root_cause"] = (
                "Function arguments or object types do not match expected usage."
            )
            analysis["suggested_next_steps"] = [
                "Check function signatures.",
                "Check constructor parameters.",
                "Check call sites.",
            ]

        elif "SyntaxError" in text:
            analysis["error_type"] = "SyntaxError"
            analysis["likely_root_cause"] = "Python syntax is invalid."
            analysis["suggested_next_steps"] = [
                "Open the file and line reported by the traceback.",
                "Fix syntax before running tests again.",
            ]

        file_matches = re.findall(r'File "([^"]+)", line (\d+)', text)
        relevant_files = []

        for file_path, line_no in file_matches:
            try:
                p = Path(file_path)
                if p.is_absolute():
                    rel = str(p.relative_to(Path(state.repo_root).resolve()))
                else:
                    rel = file_path
            except Exception:
                rel = file_path

            relevant_files.append(f"{rel}:{line_no}")

        analysis["relevant_files"] = relevant_files

        output = (
            f"Error type: {analysis['error_type']}\n"
            f"Likely root cause: {analysis['likely_root_cause']}\n"
            f"Relevant files:\n"
            + "\n".join(f"- {f}" for f in analysis["relevant_files"])
            + "\nSuggested next steps:\n"
            + "\n".join(f"- {s}" for s in analysis["suggested_next_steps"])
        )

        return ToolResult(
            tool_name="identify_error",
            success=True,
            output=output,
            metadata=analysis,
        )

    except Exception as exc:
        return ToolResult(
            tool_name="identify_error",
            success=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
        )


def git_diff(
    state: AgentState,
) -> ToolResult:
    """
    Show current git diff.
    """
    result = run_command(
        state=state,
        command="git diff -- .",
        timeout=30,
    )
    result.tool_name = "git_diff"
    return result


def retrieve_context(
    state: AgentState,
    query: str,
    top_k: int = 6,
    force_rebuild: bool = False,
) -> ToolResult:
    """
    Retrieve relevant repository context using FAISS-based RAG.
    """
    try:

        engine = _get_rag_engine(state.repo_root, force_rebuild=force_rebuild)

        if force_rebuild:
            engine.rebuild()

        results = engine.retrieve(
            query=query,
            top_k=top_k,
            vector_top_k=max(12, top_k * 2),
            keyword_top_k=max(20, top_k * 3),
        )

        context_text = engine.format_context(results)

        # Store structured context into state.
        state.retrieved_context = results

        return ToolResult(
            tool_name="retrieve_context",
            success=True,
            output=context_text,
            metadata={
                "query": query,
                "top_k": top_k,
                "num_results": len(results),
                "index_dir": ".agent_index",
            },
        )

    except Exception as exc:
        return ToolResult(
            tool_name="retrieve_context",
            success=False,
            output="",
            error=f"{type(exc).__name__}: {exc}",
            metadata={
                "query": query,
                "top_k": top_k,
            },
        )


def get_tool_map() -> Dict[str, Callable[..., ToolResult]]:
    """
    Return all executable tools for Executor.
    """
    return {
        "list_files": list_files,
        "read_file": read_file,
        "search_code": search_code,
        "retrieve_context": retrieve_context,
        "write_file": write_file,
        "replace_in_file": replace_in_file,
        "apply_patch": apply_patch,
        "run_command": run_command,
        "run_tests": run_tests,
        "identify_error": identify_error,
        "git_diff": git_diff,
    }


def execute_tool_call(
    state: AgentState,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    """
    Optional compatibility helper.
    Useful if your executor wants a single execute_tool_call entrypoint.
    """
    arguments = arguments or {}
    tool_map = get_tool_map()

    if tool_name not in tool_map:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error=f"Unknown tool: {tool_name}",
        )

    return tool_map[tool_name](state=state, **arguments)


def _validate_command_safety(command: str) -> Optional[str]:
    if not command or not command.strip():
        return "Empty command."

    for pattern in DANGEROUS_COMMAND_PATTERNS:
        if re.search(pattern, command):
            return f"Blocked dangerous command pattern: {pattern}"

    return None


def _format_command_output(stdout: str, stderr: str, exit_code: int) -> str:
    stdout = stdout or ""
    stderr = stderr or ""

    max_chars = 8000

    output = (
        f"Exit code: {exit_code}\n\n"
        f"STDOUT:\n{stdout}\n\n"
        f"STDERR:\n{stderr}"
    )

    if len(output) > max_chars:
        output = output[:max_chars] + "\n...[truncated]"

    return output


def _extract_changed_files_from_patch(patch: str) -> List[str]:
    changed = []

    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            changed.append(line.replace("+++ b/", "").strip())
        elif line.startswith("--- a/"):
            file_path = line.replace("--- a/", "").strip()
            if file_path not in changed and file_path != "/dev/null":
                changed.append(file_path)

    return sorted(set(changed))