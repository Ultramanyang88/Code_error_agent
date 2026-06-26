"""
LLM-as-judge for the coding agent eval harness.

Evaluates three dimensions per run:
  1. plan_quality   — did the agent form a sensible plan? (1-5)
  2. fix_correctness — did the agent apply the right fix? (pass/fail + reasoning)
  3. efficiency      — steps and replans relative to task difficulty (1-5)

Requires: pip install openai
Set env var: OPENAI_API_KEY
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

JUDGE_MODEL = "gpt-4o-mini"


def _call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai to use LLM-as-judge")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _parse_json_from_response(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from the LLM response."""
    import re
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"error": "failed to parse judge response", "raw": text}


def judge_plan_quality(
    task_description: str,
    plan_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Score how well-formed and sensible the agent's plan was.
    Returns: {score: 1-5, reasoning: str}
    """
    plan_text = "\n".join(
        f"Step {s.get('step_id', i+1)}: {s.get('task', '')}"
        for i, s in enumerate(plan_steps)
    )

    prompt = f"""You are evaluating a coding agent's plan for fixing a bug.

Task given to the agent:
{task_description}

Plan the agent created:
{plan_text}

Score the plan quality from 1 to 5 using these criteria:
- 5: Logical sequence, starts with inspection/error analysis, ends with validation, no redundant steps
- 4: Mostly logical, minor inefficiency (e.g. one extra step)
- 3: Partially correct but missing a key step (e.g. skips validation or reading the file first)
- 2: Wrong order or includes irrelevant steps
- 1: Completely off-track or empty

Return ONLY valid JSON with this schema:
{{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""

    raw = _call_openai(prompt)
    result = _parse_json_from_response(raw)
    result["dimension"] = "plan_quality"
    return result


def judge_fix_correctness(
    task_description: str,
    files_modified: List[str],
    trajectory: List[Dict[str, Any]],
    judge_criteria: List[str],
    validation_status: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the fix applied was correct and targeted.
    Returns: {passed: bool, reasoning: str, issues: [str]}
    """
    tool_calls = []
    for entry in trajectory:
        for tr in entry.get("tool_results", []):
            if tr.get("tool") in ("replace_in_file", "apply_patch", "write_file"):
                tool_calls.append(f"  - {tr['tool']}: success={tr['success']}")

    criteria_text = "\n".join(f"- {c}" for c in judge_criteria)
    fix_summary = "\n".join(tool_calls) if tool_calls else "No file-editing tools were called."

    prompt = f"""You are evaluating whether a coding agent correctly fixed a bug.

Task:
{task_description}

Expected fix criteria:
{criteria_text}

Files the agent modified: {files_modified}
Validation result: {validation_status}

Editing actions the agent took:
{fix_summary}

Assess whether the agent's fix meets the criteria above.
Return ONLY valid JSON with this schema:
{{"passed": <bool>, "reasoning": "<two sentences max>", "issues": ["<issue1>", "<issue2>"]}}

If validation_status is PASSED and the fix actions look targeted, passed should be true.
If the agent modified the wrong file or used a shotgun approach, passed should be false."""

    raw = _call_openai(prompt)
    result = _parse_json_from_response(raw)
    result["dimension"] = "fix_correctness"
    return result


def judge_efficiency(
    steps_completed: int,
    replan_count: int,
    total_tool_calls: int,
    elapsed_s: float,
    validation_status: str,
) -> Dict[str, Any]:
    """
    Score agent efficiency (how directly it reached the solution).
    Returns: {score: 1-5, reasoning: str}
    """
    prompt = f"""You are scoring the efficiency of a coding agent that fixed a bug.

Metrics:
- Steps completed: {steps_completed}
- Replan count: {replan_count}
- Total tool calls: {total_tool_calls}
- Elapsed time (s): {elapsed_s:.1f}
- Final validation: {validation_status}

Score efficiency from 1 to 5:
- 5: Solved in ≤5 steps, 0 replans, ≤10 tool calls
- 4: Solved in ≤8 steps, ≤1 replan
- 3: Solved with 2-3 replans or >10 tool calls
- 2: Many replans (4+) but eventually solved
- 1: Did not solve or hit budget limit

Return ONLY valid JSON with this schema:
{{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""

    raw = _call_openai(prompt)
    result = _parse_json_from_response(raw)
    result["dimension"] = "efficiency"
    return result


def run_judge(
    task_description: str,
    expected_outcome: Dict[str, Any],
    run_result: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run all three judge dimensions for one eval task.
    Returns a combined verdict dict.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "skipped": True,
            "reason": "OPENAI_API_KEY not set",
        }

    plan_steps = run_result.get("plan_steps", [])
    files_modified = run_result.get("files_modified", [])
    validation_status = run_result.get("validation_status", "NOT_RUN")
    steps_completed = run_result.get("steps_completed", 0)
    replan_count = run_result.get("replan_count", 0)
    total_tool_calls = sum(
        len(e.get("tool_results", [])) for e in trajectory
    )
    elapsed_s = run_result.get("elapsed_s", 0.0)
    judge_criteria = expected_outcome.get("judge_criteria", [])

    plan_verdict = judge_plan_quality(task_description, plan_steps)
    fix_verdict = judge_fix_correctness(
        task_description, files_modified, trajectory,
        judge_criteria, validation_status,
    )
    efficiency_verdict = judge_efficiency(
        steps_completed, replan_count, total_tool_calls,
        elapsed_s, validation_status,
    )

    overall_pass = (
        fix_verdict.get("passed", False)
        and validation_status == "PASSED"
    )

    return {
        "overall_pass": overall_pass,
        "plan_quality": plan_verdict,
        "fix_correctness": fix_verdict,
        "efficiency": efficiency_verdict,
        "summary": {
            "plan_score": plan_verdict.get("score"),
            "fix_passed": fix_verdict.get("passed"),
            "efficiency_score": efficiency_verdict.get("score"),
        },
    }
