# LLM call the tool to do the correct action

import os
from pathlib import Path
from typing import List, Dict
import subprocess

# need a doc scan tool

IGNORE_DIRS = {
    ".git", "node_modules", "dist", "build", "venv", "__pycache__",".pytest_cache", ".idea"
}

def list_files(root:str, max_entries: int = 200):
    results = []
    root_path = Path(root)
    for p in root_path.rglob("*"):
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if p.is_file():
            results.append(str(p.relative_to(root_path)))
        if len(results) >= max_entries:
            break
    return results

# search code
def search_code(root, query, max_results: int):
    results = []
    root_path = Path(root)
    for p in root_path.rglob("*"):
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue

        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            if query.lower() in line.lower():
                results.append({
                    "path": str(p.relative_to(root_path)),
                    "line": i,
                    "snippet": line.strip()
                })
            if len(results) >= max_results:
                return results
    return results

# read the file
def read_file(path: str, start_line: int = 1, end_line: int = 300):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    selected = lines[start_idx: end_idx]
    numbered = [f"{i+start_idx+1}:{line}" for i, line in enumerate(selected)]
    return "\n".join(numbered)

# write file
def write_file(path, content):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {path}"

#run command
def run_command(cmd,cwd, timeout: int = 90):
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command":cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr":proc.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as e:
        return{
            "command":cmd,
            "returncode": -1,
            "stdout": (e.stdout or "")[-8000:] if e.stdout else "",
            "stderr":f"TIMEOUT: {str(e)}",
        }
    except Exception as e:
        return{
            "command":cmd,
            "returncode": -1,
            "stdout": "",
            "stderr":repr(e),
        }

def finish(summary: str):
    return {"done": True, "summary": summary}


def choose_validation_command(repo_root: str, user_repro_cmd: str | None = None) -> str:
    root = Path(repo_root)

    if user_repro_cmd:
        return user_repro_cmd

    if (root / "pytest.ini").exists() or (root / "conftest.py").exists():
        return "pytest -q"

    if (root / "go.mod").exists():
        return "go test ./..."

    if (root / "package.json").exists():
        return "npm test"

    if (root / "pom.xml").exists():
        return "mvn test"

    if (root / "Cargo.toml").exists():
        return "cargo test"

    if (root / "requirements.txt").exists() or (root / "pyproject.toml").exists():
        return "python -m compileall ."

    if (root / "go.mod").exists():
        return "go build ./..."

    if (root / "package.json").exists():
        return "npm run build"

    return "python -m compileall ."