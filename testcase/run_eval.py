#!/usr/bin/env python3
"""
Eval harness for the coding agent.

Usage:
  # Fallback mode (no LLM, quick smoke test)
  python testcase/run_eval.py

  # LLM mode
  python testcase/run_eval.py --llm --provider openai_compatible --model qwen2.5-coder:7b

  # With LLM judge (requires OPENAI_API_KEY)
  python testcase/run_eval.py --judge

  # Run specific task only
  python testcase/run_eval.py --task task_001_division_bug

  # Save baseline for regression
  python testcase/run_eval.py --save-baseline

  # Compare against baseline
  python testcase/run_eval.py --compare-baseline
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# Make project root importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.state import ValidationStatus
from main import run_agent

TASKS_DIR = Path(__file__).parent / "tasks"
RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_PATH = Path(__file__).parent / "baseline.json"


# ── workspace setup ──────────────────────────────────────────────────────────

def setup_workspace(task_dir: Path, source_repo: Path) -> Path:
    """Copy source repo into a temp dir and overlay buggy files."""
    tmp = tempfile.mkdtemp(prefix="agent_eval_")
    work_repo = Path(tmp) / "repo"
    shutil.copytree(source_repo, work_repo)

    buggy_dir = task_dir / "buggy_files"
    if buggy_dir.exists():
        for buggy_file in buggy_dir.iterdir():
            if buggy_file.is_file():
                shutil.copy2(buggy_file, work_repo / buggy_file.name)

    return work_repo


# ── single task runner ───────────────────────────────────────────────────────

def run_task(
    task_dir: Path,
    client=None,
    use_judge: bool = False,
) -> Dict[str, Any]:
    meta_path = task_dir / "expected_outcome.json"
    desc_path = task_dir / "description.txt"

    if not meta_path.exists() or not desc_path.exists():
        return {"task_id": task_dir.name, "result": "SKIPPED", "reason": "missing files"}

    meta = json.loads(meta_path.read_text())
    description = desc_path.read_text().strip()
    task_id = meta["task_id"]

    target_rel = meta.get("target_repo", "testcase/py")
    source_repo = PROJECT_ROOT / target_rel

    work_repo = setup_workspace(task_dir, source_repo)
    trace_path = str(RESULTS_DIR / task_id / "trace.jsonl")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / task_id).mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[Task] {task_id}")
    print(f"[Repo] {work_repo}")
    print(f"[Goal] {description[:120]}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        state = run_agent(
            task_description=description,
            repo_root=str(work_repo),
            client=client,
            trace_path=trace_path,
            run_id=task_id,
        )
        elapsed = round(time.time() - t0, 2)

        validation = state.validation_status.value
        success = (validation == ValidationStatus.PASSED.value)

        plan_steps = [
            {"step_id": s.step_id, "task": s.task, "status": s.status.value}
            for s in state.plan
        ]

        run_result = {
            "task_id": task_id,
            "success": success,
            "validation_status": validation,
            "expected_validation": meta["expected_validation"],
            "run_status": state.run_status.value,
            "stop_reason": state.stop_reason,
            "replan_count": state.replan_count,
            "steps_completed": sum(1 for s in state.plan if s.status.value == "completed"),
            "steps_total": len(state.plan),
            "tool_calls_total": len(state.tool_history),
            "files_modified": state.files_modified,
            "elapsed_s": elapsed,
            "plan_steps": plan_steps,
        }

    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        run_result = {
            "task_id": task_id,
            "success": False,
            "validation_status": "ERROR",
            "expected_validation": meta["expected_validation"],
            "run_status": "error",
            "stop_reason": str(exc),
            "replan_count": 0,
            "steps_completed": 0,
            "steps_total": 0,
            "tool_calls_total": 0,
            "files_modified": [],
            "elapsed_s": elapsed,
            "plan_steps": [],
        }
    finally:
        shutil.rmtree(work_repo.parent, ignore_errors=True)

    # LLM judge
    if use_judge:
        trajectory = _load_trace(trace_path)
        try:
            from testcase.judge import run_judge
        except ImportError:
            from judge import run_judge

        print(f"[Judge] Running LLM evaluation...")
        verdict = run_judge(
            task_description=description,
            expected_outcome=meta,
            run_result=run_result,
            trajectory=trajectory,
        )
        run_result["judge"] = verdict
        _print_judge_verdict(verdict)

    # Save result
    result_file = RESULTS_DIR / task_id / "result.json"
    result_file.write_text(json.dumps(run_result, indent=2))

    status_str = "PASS" if run_result["success"] else "FAIL"
    print(f"\n[{status_str}] {task_id}")
    print(f"  validation={run_result['validation_status']} "
          f"steps={run_result['steps_completed']}/{run_result['steps_total']} "
          f"replans={run_result['replan_count']} "
          f"elapsed={run_result['elapsed_s']}s")

    return run_result


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_trace(trace_path: str) -> List[Dict[str, Any]]:
    path = Path(trace_path)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _print_judge_verdict(verdict: Dict[str, Any]) -> None:
    if verdict.get("skipped"):
        print(f"  [Judge] Skipped: {verdict.get('reason')}")
        return
    s = verdict.get("summary", {})
    print(f"  [Judge] plan={s.get('plan_score')}/5  "
          f"fix={'PASS' if s.get('fix_passed') else 'FAIL'}  "
          f"efficiency={s.get('efficiency_score')}/5")
    fix = verdict.get("fix_correctness", {})
    if fix.get("reasoning"):
        print(f"  [Judge] {fix['reasoning']}")


def _print_summary(results: List[Dict[str, Any]]) -> None:
    passed = sum(1 for r in results if r.get("success"))
    total = len(results)
    rate = 100 * passed // total if total else 0

    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY: {passed}/{total} passed ({rate}%)")
    print(f"{'='*60}")

    headers = ["Task", "Result", "Validation", "Steps", "Replans", "Time(s)"]
    rows = []
    for r in results:
        rows.append([
            r["task_id"],
            "PASS" if r["success"] else "FAIL",
            r["validation_status"],
            f"{r['steps_completed']}/{r['steps_total']}",
            str(r["replan_count"]),
            str(r["elapsed_s"]),
        ])

    col_w = [max(len(h), max((len(row[i]) for row in rows), default=0))
             for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_w))
    for row in rows:
        print(fmt.format(*row))

    if any("judge" in r for r in results):
        print(f"\nJudge scores:")
        for r in results:
            v = r.get("judge", {})
            if v and not v.get("skipped"):
                s = v.get("summary", {})
                print(f"  {r['task_id']}: "
                      f"plan={s.get('plan_score')}/5 "
                      f"fix={'PASS' if s.get('fix_passed') else 'FAIL'} "
                      f"efficiency={s.get('efficiency_score')}/5")

    avg_steps = sum(r["steps_completed"] for r in results) / max(total, 1)
    avg_replans = sum(r["replan_count"] for r in results) / max(total, 1)
    print(f"\nAverages: {avg_steps:.1f} steps/task, {avg_replans:.1f} replans/task")


def _save_baseline(results: List[Dict[str, Any]]) -> None:
    baseline = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": {r["task_id"]: r for r in results},
        "pass_rate": sum(1 for r in results if r["success"]) / max(len(results), 1),
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
    print(f"\nBaseline saved to {BASELINE_PATH}")


def _compare_baseline(results: List[Dict[str, Any]]) -> None:
    if not BASELINE_PATH.exists():
        print("No baseline found. Run with --save-baseline first.")
        return

    baseline = json.loads(BASELINE_PATH.read_text())
    base_results = baseline.get("results", {})
    regressions = []

    for r in results:
        tid = r["task_id"]
        base = base_results.get(tid, {})
        if base.get("success") and not r["success"]:
            regressions.append(f"  REGRESSION: {tid} (was PASS, now FAIL)")

    if regressions:
        print(f"\n{'!'*60}")
        print("REGRESSIONS DETECTED:")
        for reg in regressions:
            print(reg)
        print(f"{'!'*60}")
        sys.exit(1)
    else:
        base_rate = baseline.get("pass_rate", 0)
        curr_rate = sum(1 for r in results if r["success"]) / max(len(results), 1)
        print(f"\nNo regressions. Baseline={base_rate:.0%}, Current={curr_rate:.0%}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coding agent eval harness")
    p.add_argument("--llm", action="store_true", help="Use LLM client")
    p.add_argument("--provider", default="openai_compatible",
                   choices=["openai_compatible", "ollama"])
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default="qwen2.5-coder:7b")
    p.add_argument("--judge", action="store_true",
                   help="Run LLM-as-judge on each result (requires OPENAI_API_KEY)")
    p.add_argument("--task", default=None,
                   help="Run a single task by name (e.g. task_001_division_bug)")
    p.add_argument("--save-baseline", action="store_true",
                   help="Save current results as regression baseline")
    p.add_argument("--compare-baseline", action="store_true",
                   help="Compare results against saved baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    client = None
    if args.llm:
        from llm import create_local_llm_client
        client = create_local_llm_client(
            provider=args.provider,
            base_url=args.base_url,
            model=args.model,
        )
        print(f"[*] LLM mode: {args.model}")
    else:
        print("[*] Fallback mode (no LLM)")

    if not TASKS_DIR.exists():
        print(f"No tasks directory found at {TASKS_DIR}")
        sys.exit(1)

    # Discover tasks
    if args.task:
        task_dirs = [TASKS_DIR / args.task]
        if not task_dirs[0].exists():
            print(f"Task not found: {args.task}")
            sys.exit(1)
    else:
        task_dirs = sorted(d for d in TASKS_DIR.iterdir() if d.is_dir())

    print(f"[*] Running {len(task_dirs)} task(s)")

    results = []
    for task_dir in task_dirs:
        result = run_task(task_dir, client=client, use_judge=args.judge)
        results.append(result)

    _print_summary(results)

    # Save full run to results/
    run_summary_path = RESULTS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    run_summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {run_summary_path}")

    if args.save_baseline:
        _save_baseline(results)

    if args.compare_baseline:
        _compare_baseline(results)


if __name__ == "__main__":
    main()
