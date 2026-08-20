#!/usr/bin/env python3
"""
Eval harness for the 30-task portfolio manifest (testcase/eval_sets/
manifests/portfolio_30.json), built by testcase/build_eval_set.py from
QuixBugs, BugsInPy, and SWE-bench Lite.

Separate from testcase/run_eval.py on purpose: that one runs the original
3-task hand-rolled fixture set (testcase/tasks/*/, all sharing one target
repo) and stays a fast, dependency-free smoke test for the agent loop
itself. This one runs real external tasks -- network access, per-source
setup that can take minutes (bugsinpy-compile builds a fresh venv per bug),
and a grader that can legitimately fail to even set up an environment
(setup_failed) independently of whether the agent's fix was any good. Keep
them separate rather than merging: a broken swebench dependency shouldn't
be able to fail the quick sanity check that run_eval.py is for.

Usage:
  # Fallback mode (no LLM), QuixBugs only -- fastest way to sanity-check the wiring
  python testcase/run_portfolio_eval.py --source quixbugs

  # LLM mode, everything
  python testcase/run_portfolio_eval.py --llm --provider openai_compatible --model qwen2.5-coder:7b

  # Single instance
  python testcase/run_portfolio_eval.py --instance quixbugs__max_sublist_sum

  # Save / compare regression baseline (separate from run_eval.py's baseline.json)
  python testcase/run_portfolio_eval.py --save-baseline
  python testcase/run_portfolio_eval.py --compare-baseline
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import run_agent
from core.state import ValidationStatus
from testcase.eval_sets import workspace as eval_workspace

MANIFEST_PATH = Path(__file__).parent / "eval_sets" / "manifests" / "portfolio_30.json"
WORKSPACES_DIR = Path(__file__).parent / "eval_sets" / "workspaces"
RESULTS_DIR = Path(__file__).parent / "results" / "portfolio"
BASELINE_PATH = Path(__file__).parent / "eval_sets" / "baseline.json"


def load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH}. Run testcase/build_eval_set.py first.")
        sys.exit(1)
    return json.loads(MANIFEST_PATH.read_text())


def run_instance(instance: Dict[str, Any], client=None) -> Dict[str, Any]:
    instance_id = instance["instance_id"]
    source = instance["source"]

    result_dir = RESULTS_DIR / instance_id
    result_dir.mkdir(parents=True, exist_ok=True)
    ws_dest = WORKSPACES_DIR / instance_id

    print(f"\n{'='*60}")
    print(f"[{source}] {instance_id}")
    print(f"[Goal] {instance['problem_statement'][:120].splitlines()[0]}")
    print(f"{'='*60}")

    t0 = time.time()
    outcome: Dict[str, Any] = {
        "instance_id": instance_id,
        "source": source,
        "category": instance.get("category"),
        "difficulty": instance.get("difficulty"),
    }

    try:
        # prepare_workspace's return value is the actual workspace root --
        # NOT necessarily ws_dest itself. bugsinpy-checkout, for instance,
        # clones one level down into ws_dest/<project>/, so the real
        # workspace to point the agent and the grader at is that
        # subdirectory, not the container dir we asked it to use.
        ws_dest = eval_workspace.prepare_workspace(instance, ws_dest)
    except Exception as exc:
        outcome.update({
            "success": False, "setup_failed": True, "stage": "workspace_prep",
            "detail": str(exc), "elapsed_s": round(time.time() - t0, 2),
        })
        print(f"[SETUP FAILED] {instance_id}: {exc}")
        _write_result(result_dir, outcome)
        return outcome

    trace_path = str(result_dir / "trace.jsonl")
    try:
        state = run_agent(
            task_description=instance["problem_statement"],
            repo_root=str(ws_dest),
            client=client,
            trace_path=trace_path,
            run_id=instance_id,
        )
        agent_elapsed = round(time.time() - t0, 2)
        outcome["agent_stop_reason"] = state.stop_reason
        outcome["agent_replan_count"] = state.replan_count
        outcome["agent_steps_completed"] = sum(1 for s in state.plan if s.status.value == "completed")
        outcome["agent_files_modified"] = state.files_modified
        outcome["agent_validation_status"] = state.validation_status.value
    except Exception as exc:
        outcome.update({
            "success": False, "setup_failed": False, "stage": "agent_run",
            "detail": str(exc), "elapsed_s": round(time.time() - t0, 2),
        })
        print(f"[AGENT ERROR] {instance_id}: {exc}")
        _write_result(result_dir, outcome)
        return outcome

    try:
        grade = eval_workspace.grade(instance, ws_dest)
    except Exception as exc:
        outcome.update({
            "success": False, "setup_failed": True, "stage": "grading",
            "detail": str(exc), "elapsed_s": round(time.time() - t0, 2),
        })
        print(f"[GRADE ERROR] {instance_id}: {exc}")
        _write_result(result_dir, outcome)
        return outcome

    elapsed = round(time.time() - t0, 2)
    outcome.update({
        "success": grade.passed,
        "setup_failed": grade.setup_failed,
        "stage": "graded",
        "detail": grade.detail,
        "elapsed_s": elapsed,
    })

    status_str = "SETUP_FAILED" if grade.setup_failed else ("PASS" if grade.passed else "FAIL")
    print(f"\n[{status_str}] {instance_id}  elapsed={elapsed}s  "
          f"steps={outcome['agent_steps_completed']} replans={outcome['agent_replan_count']}")

    _write_result(result_dir, outcome)
    return outcome


def _write_result(result_dir: Path, outcome: Dict[str, Any]) -> None:
    (result_dir / "result.json").write_text(json.dumps(outcome, indent=2, default=str))


# ── reporting ────────────────────────────────────────────────────────────

def _print_summary(results: List[Dict[str, Any]]) -> None:
    graded = [r for r in results if not r.get("setup_failed")]
    setup_failed = [r for r in results if r.get("setup_failed")]
    passed = sum(1 for r in graded if r.get("success"))
    total = len(results)

    print(f"\n{'='*60}")
    print(f"PORTFOLIO EVAL SUMMARY: {passed}/{len(graded)} graded tasks passed "
          f"({total - len(graded)} setup failures excluded from pass rate)")
    print(f"{'='*60}")

    headers = ["Instance", "Source", "Result", "Steps", "Replans", "Time(s)"]
    rows = []
    for r in results:
        status = "SETUP_FAIL" if r.get("setup_failed") else ("PASS" if r.get("success") else "FAIL")
        rows.append([
            r["instance_id"], r["source"], status,
            str(r.get("agent_steps_completed", "-")),
            str(r.get("agent_replan_count", "-")),
            str(r.get("elapsed_s", "-")),
        ])

    col_w = [max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_w))
    for row in rows:
        print(fmt.format(*row))

    if setup_failed:
        print(f"\n{len(setup_failed)} setup failure(s) (environment issue, not a graded result):")
        for r in setup_failed:
            print(f"  {r['instance_id']}: {r.get('detail', '')[:150]}")


def _save_baseline(results: List[Dict[str, Any]]) -> None:
    baseline = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": {r["instance_id"]: r for r in results},
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, default=str))
    print(f"\nBaseline saved to {BASELINE_PATH}")


def _compare_baseline(results: List[Dict[str, Any]]) -> None:
    if not BASELINE_PATH.exists():
        print("No baseline found. Run with --save-baseline first.")
        return

    baseline = json.loads(BASELINE_PATH.read_text())
    base_results = baseline.get("results", {})
    regressions = []
    for r in results:
        base = base_results.get(r["instance_id"], {})
        # A prior setup_failed carries no signal either way -- only compare
        # when the baseline run actually graded the instance.
        if base.get("success") and not base.get("setup_failed") and not r.get("setup_failed") and not r["success"]:
            regressions.append(f"  REGRESSION: {r['instance_id']} (was PASS, now FAIL)")

    if regressions:
        print(f"\n{'!'*60}\nREGRESSIONS DETECTED:")
        for reg in regressions:
            print(reg)
        print("!" * 60)
        sys.exit(1)
    print("\nNo regressions.")


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Portfolio (QuixBugs/BugsInPy/SWE-bench) eval harness")
    p.add_argument("--llm", action="store_true", help="Use LLM client")
    p.add_argument("--provider", default="openai_compatible", choices=["openai_compatible", "ollama"])
    # None (not a hardcoded string) so create_local_llm_client() falls
    # through to $LLM_BASE_URL/$LLM_MODEL when these flags aren't given --
    # see llm.py. Pass either flag explicitly to override .env for one run.
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--source", choices=["quixbugs", "bugsinpy", "swebench"], default=None,
                   help="Only run instances from this source")
    p.add_argument("--instance", default=None, help="Run a single instance by instance_id")
    p.add_argument("--limit", type=int, default=None, help="Cap the number of instances run")
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--compare-baseline", action="store_true")
    p.add_argument("--keep-workspaces", action="store_true",
                   help="Don't delete testcase/eval_sets/workspaces/ entries after each run")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    client = None
    if args.llm:
        from llm import create_local_llm_client
        client = create_local_llm_client(provider=args.provider, base_url=args.base_url, model=args.model)
        # client.model/base_url, not args.model/base_url -- those are often
        # None here (resolved from $LLM_BASE_URL/$LLM_MODEL inside
        # create_local_llm_client when the flags aren't given), and printing
        # the unresolved None would be actively misleading about what this
        # run is actually about to call.
        print(f"[*] LLM mode: {client.model} @ {client.base_url}")
    else:
        print("[*] Fallback mode (no LLM)")

    manifest = load_manifest()
    instances = manifest["instances"]

    if args.instance:
        instances = [i for i in instances if i["instance_id"] == args.instance]
        if not instances:
            print(f"Instance not found: {args.instance}")
            sys.exit(1)
    if args.source:
        instances = [i for i in instances if i["source"] == args.source]
    if args.limit:
        instances = instances[:args.limit]

    if not instances:
        print("No instances matched the given filters.")
        sys.exit(1)

    print(f"[*] Running {len(instances)} instance(s) from {MANIFEST_PATH.name}")

    results = []
    for instance in instances:
        result = run_instance(instance, client=client)
        results.append(result)
        if not args.keep_workspaces:
            shutil.rmtree(WORKSPACES_DIR / instance["instance_id"], ignore_errors=True)

    _print_summary(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_summary_path = RESULTS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    run_summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results saved to {run_summary_path}")

    if args.save_baseline:
        _save_baseline(results)
    if args.compare_baseline:
        _compare_baseline(results)


if __name__ == "__main__":
    main()
