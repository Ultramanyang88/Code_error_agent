#!/usr/bin/env python3
"""Build a deterministic, diverse manifest of external coding-agent tasks."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = PROJECT_ROOT / "testcase" / "eval_sets"
DEFAULT_CACHE_DIR = EVAL_ROOT / "cache"
DEFAULT_OUTPUT = EVAL_ROOT / "manifests" / "portfolio_30.json"

QUIXBUGS_URL = "https://github.com/jkoppel/QuixBugs.git"
BUGSINPY_URL = "https://github.com/soarsmu/BugsInPy.git"
DEFAULT_SWEBENCH_DATASET = "SWE-bench/SWE-bench_Lite"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(command)
    print(f"[run] {printable}")
    subprocess.run(command, cwd=cwd, check=True)


def ensure_git_repo(url: str, destination: Path, refresh: bool) -> None:
    if (destination / ".git").is_dir():
        if refresh:
            run(["git", "pull", "--ff-only"], cwd=destination)
        else:
            print(f"[cache] Reusing {destination}")
        return

    if destination.exists():
        raise RuntimeError(
            f"Cache path exists but is not a Git repository: {destination}. "
            "Move it aside or rerun with a different --cache-dir."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", url, str(destination)])


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in read_text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def changed_lines(patch: str) -> int:
    return sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


def changed_files(patch: str) -> int:
    files = set(re.findall(r"^diff --git a/(.+?) b/", patch, flags=re.MULTILINE))
    if files:
        return len(files)
    return 1 if patch.strip() else 0


def infer_difficulty(patch: str) -> str:
    files = changed_files(patch)
    lines = changed_lines(patch)
    if files <= 1 and lines <= 20:
        return "easy"
    if files <= 3 and lines <= 100:
        return "medium"
    return "hard"


def infer_category(text: str, patch: str = "") -> str:
    haystack = f"{text}\n{patch}".lower()
    keyword_groups = [
        ("concurrency", ("race condition", "deadlock", "thread", "async", "lock", "concurr")),
        ("dependency_import", ("import", "module", "dependency", "package", "version")),
        ("exception_handling", ("exception", "traceback", "raise ", "error handling", "try:", "except")),
        ("parsing_validation", ("parse", "token", "serializ", "deserializ", "validation", "invalid input")),
        ("configuration_build", ("config", "environment", "docker", "build", "install", "command line")),
        ("state_cache", ("cache", "session", "queue", "state", "memory", "persist")),
        ("api_interface", ("api", "endpoint", "request", "response", "interface", "signature")),
        ("boundary_condition", ("off-by-one", "index", "empty", "none", "null", "zero", "boundary")),
    ]
    for category, keywords in keyword_groups:
        if any(keyword in haystack for keyword in keywords):
            return category

    removed = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    if removed and added:
        comparison_tokens = ("==", "!=", "<=", ">=", "<", ">")
        if any(token in removed or token in added for token in comparison_tokens):
            return "comparison_logic"
        if re.search(r"\b\d+\b", removed) and re.search(r"\b\d+\b", added):
            return "constant_off_by_one"
    return "general_logic"


def make_unified_diff(buggy: Path, fixed: Path) -> str:
    return "".join(
        difflib.unified_diff(
            read_text(buggy).splitlines(keepends=True),
            read_text(fixed).splitlines(keepends=True),
            fromfile=str(buggy.name),
            tofile=str(fixed.name),
        )
    )


def balanced_sample(
    rows: Iterable[dict[str, Any]],
    count: int,
    seed: int,
    group_key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    candidates = sorted(rows, key=lambda row: row["instance_id"])
    if count < 0:
        raise ValueError("Task counts must be non-negative")
    if len(candidates) < count:
        raise RuntimeError(f"Requested {count} tasks, but only {len(candidates)} are available")

    rng = random.Random(seed)
    tie_breakers = {row["instance_id"]: rng.random() for row in candidates}
    selected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()

    while len(selected) < count:
        best = min(
            candidates,
            key=lambda row: (
                source_counts[group_key(row)],
                category_counts[row.get("category", "unknown")],
                difficulty_counts[row.get("difficulty", "unknown")],
                tie_breakers[row["instance_id"]],
            ),
        )
        candidates.remove(best)
        selected.append(best)
        source_counts[group_key(best)] += 1
        category_counts[best.get("category", "unknown")] += 1
        difficulty_counts[best.get("difficulty", "unknown")] += 1

    return selected


def load_quixbugs(cache_dir: Path, count: int, seed: int, refresh: bool) -> list[dict[str, Any]]:
    repo = cache_dir / "quixbugs"
    ensure_git_repo(QUIXBUGS_URL, repo, refresh)

    rows: list[dict[str, Any]] = []
    for buggy in sorted((repo / "python_programs").glob("*.py")):
        name = buggy.stem
        fixed = repo / "correct_python_programs" / buggy.name
        test = repo / "python_testcases" / f"test_{name}.py"
        if name.startswith("__") or not fixed.exists() or not test.exists():
            continue
        patch = make_unified_diff(buggy, fixed)
        rows.append(
            {
                "source": "quixbugs",
                "instance_id": f"quixbugs__{name}",
                "language": "python",
                "repository": "jkoppel/QuixBugs",
                "problem_statement": (
                    f"Fix the defect in python_programs/{buggy.name}. "
                    f"The tests in python_testcases/{test.name} must pass."
                ),
                "category": infer_category(name.replace("_", " "), patch),
                "difficulty": infer_difficulty(patch),
                "workspace": {
                    "strategy": "quixbugs_isolated",
                    "cache_path": "cache/quixbugs",
                    "target_file": f"python_programs/{buggy.name}",
                    "test_file": f"python_testcases/{test.name}",
                    "exclude": ["correct_python_programs", "correct_java_programs"],
                },
                "grading": {
                    "command": f"python -m pytest python_testcases/{test.name} -q",
                },
            }
        )

    return balanced_sample(rows, count, seed, group_key=lambda row: row["category"])


def load_bugsinpy(cache_dir: Path, count: int, seed: int, refresh: bool) -> list[dict[str, Any]]:
    repo = cache_dir / "bugsinpy"
    ensure_git_repo(BUGSINPY_URL, repo, refresh)

    rows: list[dict[str, Any]] = []
    for bug_dir in sorted((repo / "projects").glob("*/bugs/*")):
        if not bug_dir.is_dir():
            continue
        project = bug_dir.parent.parent.name
        bug_id = bug_dir.name
        info_path = bug_dir / "bug.info"
        patch_path = bug_dir / "bug_patch.txt"
        if not info_path.exists() or not patch_path.exists():
            continue
        info = parse_key_value_file(info_path)
        patch = read_text(patch_path)
        issue_url = next((value for key, value in info.items() if "issue" in key.lower() and value), None)
        rows.append(
            {
                "source": "bugsinpy",
                "instance_id": f"bugsinpy__{project}__{bug_id}",
                "language": "python",
                "repository": project,
                "problem_statement": (
                    f"Fix BugsInPy bug {bug_id} in the {project} project so its triggering tests pass."
                ),
                "issue_url": issue_url,
                "category": infer_category(f"{project} {json.dumps(info, sort_keys=True)}", patch),
                "difficulty": infer_difficulty(patch),
                "workspace": {
                    "strategy": "bugsinpy_checkout",
                    "cache_path": "cache/bugsinpy",
                    "project": project,
                    "bug_id": bug_id,
                },
                "grading": {
                    "framework": "bugsinpy",
                    "commands": ["bugsinpy-compile", "bugsinpy-test"],
                },
            }
        )

    return balanced_sample(rows, count, seed, group_key=lambda row: row["repository"])


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def load_swebench(dataset_name: str, count: int, seed: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The optional 'datasets' package is required for SWE-bench. "
            "Run: pip install -r testcase/requirements-eval.txt"
        ) from exc

    print(f"[dataset] Loading {dataset_name} test split")
    dataset = load_dataset(dataset_name, split="test")
    rows: list[dict[str, Any]] = []
    for item in dataset:
        patch = str(item.get("patch") or "")
        problem = str(item.get("problem_statement") or "").strip()
        repo = str(item.get("repo") or "unknown")
        instance_id = str(item["instance_id"])
        rows.append(
            {
                "source": "swebench",
                "instance_id": instance_id,
                "language": "python",
                "repository": repo,
                "problem_statement": problem,
                "issue_url": item.get("issue_url"),
                "category": infer_category(problem, patch),
                "difficulty": infer_difficulty(patch),
                "workspace": {
                    "strategy": "git_base_commit",
                    "repository": repo,
                    "base_commit": item.get("base_commit"),
                    "version": item.get("version"),
                },
                "grading": {
                    "framework": "swebench",
                    "dataset": dataset_name,
                    "instance_id": instance_id,
                    "fail_to_pass": parse_json_list(item.get("FAIL_TO_PASS")),
                    "pass_to_pass": parse_json_list(item.get("PASS_TO_PASS")),
                },
            }
        )

    return balanced_sample(rows, count, seed, group_key=lambda row: row["repository"])


def summarize(instances: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "by_source": dict(sorted(Counter(row["source"] for row in instances).items())),
        "by_category": dict(sorted(Counter(row["category"] for row in instances).items())),
        "by_difficulty": dict(sorted(Counter(row["difficulty"] for row in instances).items())),
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    counts = (args.quixbugs_count, args.bugsinpy_count, args.swebench_count)
    if any(count < 0 for count in counts):
        raise ValueError("Task counts must be non-negative")
    if sum(counts) == 0:
        raise ValueError("At least one source count must be greater than zero")

    cache_dir = args.cache_dir.resolve()
    instances: list[dict[str, Any]] = []
    if args.quixbugs_count:
        instances.extend(load_quixbugs(cache_dir, args.quixbugs_count, args.seed, args.refresh))
    if args.bugsinpy_count:
        instances.extend(load_bugsinpy(cache_dir, args.bugsinpy_count, args.seed + 1, args.refresh))
    if args.swebench_count:
        instances.extend(load_swebench(args.swebench_dataset, args.swebench_count, args.seed + 2))

    return {
        "schema_version": 1,
        "name": args.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sources": {
            "quixbugs": QUIXBUGS_URL,
            "bugsinpy": BUGSINPY_URL,
            "swebench": args.swebench_dataset,
        },
        "summary": summarize(instances),
        "instances": instances,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic mixed evaluation-set manifest."
    )
    parser.add_argument("--name", default="portfolio-30-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quixbugs-count", type=int, default=10)
    parser.add_argument("--bugsinpy-count", type=int, default=10)
    parser.add_argument("--swebench-count", type=int, default=10)
    parser.add_argument("--swebench-dataset", default=DEFAULT_SWEBENCH_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fast-forward cached Git metadata repositories before selecting tasks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"[done] Wrote {len(manifest['instances'])} tasks to {args.output}")
    for dimension, counts in manifest["summary"].items():
        print(f"[summary] {dimension}: {counts}")


if __name__ == "__main__":
    main()
