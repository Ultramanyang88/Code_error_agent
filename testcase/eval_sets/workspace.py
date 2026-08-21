"""
Per-source workspace preparation + grading for testcase/eval_sets/manifests/
portfolio_30.json instances.

The manifest (testcase/build_eval_set.py) is deliberately just a catalog --
it names *which* tasks to run and how to grade them, but doesn't materialize
runnable workspaces or apply reference patches (see eval_sets/README.md's
"Manifest Safety" section for why). This module is the missing piece: given
one manifest instance, prepare_workspace() builds an isolated directory the
agent can be pointed at, and grade() checks whether the agent's edits made
the instance's tests pass.

Three sources, three very different mechanisms:
  - quixbugs:  self-contained, pure Python, no external deps. Copy the
    relevant files into an isolated dir and run pytest directly.
  - bugsinpy:  drives BugsInPy's own bash framework (bugsinpy-checkout /
    -compile / -test), which clones the real historical project and builds
    a fresh venv per bug. Real network + real (sometimes old) dependencies,
    so setup can fail independently of whether the agent's fix is correct --
    grade() reports that as setup_failed, not passed=False, so a broken
    19-year-old numpy pin doesn't silently read as "the agent failed".
  - swebench:  SIMPLIFIED grading, not the official SWE-bench Docker
    harness. Clones the real repo at base_commit and tries a generic
    `pip install -e .` in a fresh venv built just for that one workspace
    (never the interpreter running this harness -- see
    _isolated_venv_python()'s docstring for why that distinction is load-
    bearing, not just tidiness), then runs the fail_to_pass/pass_to_pass
    test lists directly. This is lower-fidelity than the official
    per-instance Docker images (which exist precisely because dependency/
    Python-version requirements vary per instance) -- expect a meaningful
    fraction of instances to fail at the setup step on version mismatches
    alone. That tradeoff was chosen deliberately to avoid a Docker-image-
    per-task footprint; see the grading README.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

EVAL_ROOT = Path(__file__).resolve().parent
CACHE_DIR = EVAL_ROOT / "cache"


@dataclass
class GradeResult:
    """
    passed: whether the instance's tests came back green.
    setup_failed: environment/checkout/install never got far enough to run
        the tests at all -- distinct from passed=False (tests ran and
        failed) so a broken dependency doesn't read as "the agent's fix is
        wrong". Callers should report this as a third bucket, not a FAIL.
    detail: captured stdout/stderr, truncated, for a human to skim.
    """
    passed: bool
    setup_failed: bool = False
    detail: str = ""


def _truncate(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def _run(command: list, *, cwd: Optional[Path] = None, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)


_bash_path_cache: Optional[str] = None


def _find_bash4() -> Optional[str]:
    """
    bugsinpy-test uses `&>>` redirections throughout (see
    framework/bin/bugsinpy-test), a bash-4+ syntax -- macOS's stock
    /bin/bash is 3.2.57 (unchanged since ~2007, frozen pre-GPLv3) and
    can't parse it, failing with a confusing "syntax error near unexpected
    token `>'" deep inside the script rather than anything that points at
    the real cause. Verified live on this machine. Prefer a Homebrew bash
    if one's installed; fall back to whatever `bash` resolves to on PATH
    and let the caller decide whether its version is good enough.
    """
    global _bash_path_cache
    if _bash_path_cache is not None:
        return _bash_path_cache or None

    candidates = ["/opt/homebrew/bin/bash", "/usr/local/bin/bash", shutil.which("bash") or "bash"]
    for candidate in candidates:
        if not Path(candidate).exists() and candidate not in ("bash",):
            continue
        try:
            proc = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
        first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
        # "GNU bash, version 5.2.21(1)-release ..." / "... 3.2.57(1)-release ..."
        match = re.search(r"version (\d+)\.", first_line)
        if match and int(match.group(1)) >= 4:
            _bash_path_cache = candidate
            return candidate

    _bash_path_cache = ""
    return None


# ── quixbugs ─────────────────────────────────────────────────────────────

def _prepare_quixbugs(instance: Dict[str, Any], dest: Path) -> Path:
    repo = CACHE_DIR / "quixbugs"
    if not repo.is_dir():
        raise RuntimeError(f"QuixBugs cache missing at {repo} -- run build_eval_set.py first")

    # A stale workspace from an interrupted previous run (killed mid-run,
    # crashed, etc.) blocks shutil.copytree() below with FileExistsError --
    # verified live, that's exactly what happened to a run killed partway
    # through. bugsinpy/swebench's prepare functions already clear `dest`
    # first for the same reason; this one was missing it.
    if dest.exists():
        shutil.rmtree(dest)

    # Allowlist copy, not "copy everything then exclude": correct_python_programs/
    # and correct_java_programs/ (the reference fixes) are simply never
    # touched, so there's no exclude list to keep in sync or get wrong.
    # python_testcases/ carries the test files AND their shared helpers
    # (load_testdata.py, node.py); json_testcases/ is the test data those
    # helpers read; conftest.py (repo root) defines the pytest.use_correct
    # toggle the test files import against -- all three are required for
    # `pytest <test_file>` to even collect, not just to grade meaningfully.
    shutil.copytree(repo / "python_programs", dest / "python_programs")
    shutil.copytree(repo / "python_testcases", dest / "python_testcases")
    shutil.copytree(repo / "json_testcases", dest / "json_testcases")
    shutil.copy2(repo / "conftest.py", dest / "conftest.py")
    return dest


def _grade_quixbugs(instance: Dict[str, Any], workspace_dir: Path) -> GradeResult:
    test_file = instance["workspace"]["test_file"]
    # No --correct: default pytest.use_correct=False means the test imports
    # from python_programs/ -- the file the agent just edited. Grading the
    # fixed reference implementation is never what we want here.
    proc = _run([sys.executable, "-m", "pytest", test_file, "-q"], cwd=workspace_dir, timeout=60)
    output = proc.stdout + proc.stderr
    return GradeResult(passed=proc.returncode == 0, detail=_truncate(output))


# ── bugsinpy ─────────────────────────────────────────────────────────────

_BUGSINPY_FAIL_MARKERS = ("failed", "error", "traceback (most recent call last)")
_BUGSINPY_PASS_MARKERS = ("passed", "ok\n", " ok")


def _bugsinpy_bin(name: str) -> Path:
    return CACHE_DIR / "bugsinpy" / "framework" / "bin" / name


def _prepare_bugsinpy(instance: Dict[str, Any], dest: Path) -> Path:
    ws = instance["workspace"]
    project = ws["project"]
    checkout = _bugsinpy_bin("bugsinpy-checkout")
    if not checkout.exists():
        raise RuntimeError(f"BugsInPy framework missing at {checkout} -- run build_eval_set.py first")

    # dest must not pre-exist -- bugsinpy-checkout treats a non-empty,
    # previously-unused directory as an error, and creates it itself.
    if dest.exists():
        shutil.rmtree(dest)

    # -v 0: buggy version (the agent's starting point). -v 1 would check
    # out the already-fixed version, which defeats the point of an eval.
    #
    # bugsinpy-checkout's own script does `git clone $githubURL
    # "$work_dir/$project_name"` -- the actual checkout (and the
    # bugsinpy_bug.info/bugsinpy_requirements.txt/bugsinpy_run_test.sh
    # marker files bugsinpy-compile/-test require) lands one level down at
    # dest/<project>/, not at dest/ itself. Verified by tracing the real
    # framework script, not assumed -- an earlier version of this function
    # treated `dest` as the workspace and failed cleanly (setup_failed) on
    # every real bugsinpy instance because of it.
    # Prefer a modern bash if available, same as _grade_bugsinpy: checkout
    # itself doesn't need bash 4, but per-project bugsinpy_setup.sh scripts
    # (copied in from the project's own bug metadata) are out of our
    # control and could hit the same &>> issue for some other project.
    bash = _find_bash4() or "bash"
    proc = _run(
        [bash, str(checkout), "-p", project, "-i", ws["bug_id"], "-v", "0", "-w", str(dest)],
        timeout=300,
    )
    project_dir = dest / project
    if not (project_dir / "bugsinpy_bug.info").exists():
        raise RuntimeError(
            f"bugsinpy-checkout did not produce a valid checkout for "
            f"{project}#{ws['bug_id']}: {proc.stdout}\n{proc.stderr}"
        )
    return project_dir


# Deliberately only "passed"/"failed", not "error"/"skipped"/"deselected":
# pytest reports collection-time crashes as e.g. "1 error in 0.03s" too --
# verified live, that line alone would otherwise pass as "a real pytest run
# happened" for a case that's actually a bare ModuleNotFoundError during
# import, before a single test ran. Only passed/failed counts are reliable
# evidence real tests were actually collected and executed; a summary
# containing only error/skipped is treated the same as no summary at all.
_PYTEST_SUMMARY_RE = re.compile(r"\d+\s+(passed|failed)\b", re.IGNORECASE)
_UNITTEST_RAN_RE = re.compile(r"^Ran \d+ tests?", re.MULTILINE)


def _classify_bugsinpy_output(output: str) -> "tuple[bool, bool]":
    """
    Returns (passed, setup_failed). Best-effort text parse, not an exit
    code: bugsinpy-test's own script captures pytest/unittest output into
    `res_first=$(...)` without propagating its exit status, so a clean
    process exit is not a reliable signal here (see
    framework/bin/bugsinpy-test).

    First checks whether pytest/unittest produced a real summary line at
    all ("N passed/failed/error...", or unittest's "Ran N tests"). If it
    did, trust the pass/fail markers as a normal graded result. If it
    didn't -- e.g. pytest itself crashed during its own startup/import,
    which is exactly what an old bugsinpy_requirements.txt pin can do
    against a venv built with whatever `python3` happens to resolve to on
    the host today (verified live: BugsInPy's ancient `py`/`pytest` pins
    crash with AttributeError importing their own vendored apipkg shim
    under Python 3.13) -- that's an environment problem, not a graded
    loss, so it's reported as setup_failed rather than a false FAIL.
    """
    has_summary = bool(_PYTEST_SUMMARY_RE.search(output) or _UNITTEST_RAN_RE.search(output))
    lowered = output.lower()
    has_fail_marker = any(marker in lowered for marker in _BUGSINPY_FAIL_MARKERS)

    if has_summary:
        if has_fail_marker:
            return False, False
        return any(marker in lowered for marker in _BUGSINPY_PASS_MARKERS), False

    # No summary line at all: the test runner never got far enough to
    # report anything real. A traceback/error with no summary is treated
    # as a broken environment; truly empty/unrecognizable output falls
    # back to the old safe default (failed, not passed, not misreported
    # as a clean setup either -- there's no clear signal it's an
    # environment issue specifically).
    return False, has_fail_marker


def _grade_bugsinpy(instance: Dict[str, Any], workspace_dir: Path) -> GradeResult:
    compile_script = _bugsinpy_bin("bugsinpy-compile")
    test_script = _bugsinpy_bin("bugsinpy-test")

    # Builds a fresh venv + installs bugsinpy_requirements.txt -- can be
    # slow, and can fail outright for old bugs whose pinned deps no longer
    # build on a modern Python/OS. That's a setup failure, not a graded
    # fail: the agent never got a chance to run.
    #
    # cwd=workspace_dir is required here even though -w is also passed:
    # verified by tracing the actual script -- bugsinpy-compile validates
    # $work_dir/bugsinpy_bug.info etc. but never `cd`s into $work_dir before
    # `python3 -m venv env`, so without this the venv gets built relative to
    # whatever directory this process happened to be launched from instead
    # of the workspace (confirmed live: it built a stray env/ at the repo
    # root the first time this ran without cwd set).
    compile_proc = _run(
        [_find_bash4() or "bash", str(compile_script), "-w", str(workspace_dir)], cwd=workspace_dir, timeout=900,
    )
    if compile_proc.returncode != 0:
        return GradeResult(
            passed=False, setup_failed=True,
            detail=_truncate("bugsinpy-compile failed:\n" + compile_proc.stdout + compile_proc.stderr),
        )

    # bugsinpy-test uses `&>>` redirections throughout, which needs bash 4+
    # -- macOS's stock /bin/bash (3.2.57) can't parse that and dies with a
    # confusing "syntax error near unexpected token `>'" that looks nothing
    # like a version problem (verified live). Use a modern bash if one's
    # installed (e.g. `brew install bash`); fail clearly, as setup_failed,
    # if not -- not a garbled bash parser error, and not a false FAIL.
    bash = _find_bash4()
    if bash is None:
        return GradeResult(
            passed=False, setup_failed=True,
            detail=(
                "bugsinpy-test requires bash >= 4 (uses `&>>` redirections); "
                "only bash 3.2 (macOS's stock /bin/bash) was found. "
                "Install a modern bash, e.g. `brew install bash`, and re-run."
            ),
        )

    # "-r" (run only the bug's relevant tests) must be the FIRST argument,
    # not just anywhere in the flags -- bugsinpy-test only recognizes -a/-r
    # via a `case $1 in ...` check on the very first positional arg, before
    # its getopts loop even starts (getopts itself doesn't know about -r at
    # all and will warn "illegal option" regardless of position; that
    # warning is harmless, but -r landing anywhere but $1 means neither
    # run_all nor relevant ever gets set and the script falls through to an
    # unintended branch -- verified live, that's what produced the bash
    # syntax error above before this was traced to the real bash-4 cause).
    test_proc = _run([bash, str(test_script), "-r", "-w", str(workspace_dir)], cwd=workspace_dir, timeout=300)
    output = test_proc.stdout + test_proc.stderr
    passed, setup_failed = _classify_bugsinpy_output(output)
    return GradeResult(passed=passed, setup_failed=setup_failed, detail=_truncate(output))


# ── swebench (simplified, no Docker) ────────────────────────────────────

def _prepare_swebench(instance: Dict[str, Any], dest: Path) -> Path:
    ws = instance["workspace"]
    repo_url = f"https://github.com/{ws['repository']}.git"
    if dest.exists():
        shutil.rmtree(dest)

    # Full clone (not --depth 1): base_commit is an arbitrary historical
    # commit, not necessarily reachable from a shallow clone of the
    # default branch tip.
    clone_proc = _run(["git", "clone", repo_url, str(dest)], timeout=600)
    if clone_proc.returncode != 0:
        raise RuntimeError(f"git clone failed for {repo_url}: {clone_proc.stderr}")

    checkout_proc = _run(["git", "checkout", ws["base_commit"]], cwd=dest, timeout=60)
    if checkout_proc.returncode != 0:
        raise RuntimeError(f"git checkout {ws['base_commit']} failed: {checkout_proc.stderr}")
    return dest


def _isolated_venv_python(workspace_dir: Path) -> str:
    """
    Build (if needed) and return the python executable of a venv scoped to
    this one workspace, for swebench's install+test step.

    MUST NOT use sys.executable here: an earlier version of this function
    ran `pip install -e .` with sys.executable directly, which installs
    into *that interpreter's* site-packages -- verified live, and the hard
    way. Grading the pytest-dev/pytest instance installed that historical
    pytest checkout as an editable package directly into this project's own
    .venv, silently replacing the real pytest with an editable link into a
    workspace directory that gets deleted after the run. Every other
    project command that shells out to "pytest" (including this project's
    own `pytest testcase/test_all.py`) broke with ModuleNotFoundError until
    it was manually repaired -- a handful of other swebench instances
    (django, flask, sphinx, sympy, xarray) did the same thing to their own
    package names. A per-instance venv -- same idea as BugsInPy's
    bugsinpy-compile building one per bug -- contains an install to a
    directory that's thrown away with the workspace, never the interpreter
    running this harness.
    """
    venv_dir = workspace_dir / ".agent_eval_venv"
    venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv_python.exists():
        proc = _run([sys.executable, "-m", "venv", str(venv_dir)], timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create isolated venv: {proc.stderr}")
    return str(venv_python)


def _grade_swebench(instance: Dict[str, Any], workspace_dir: Path) -> GradeResult:
    grading = instance["grading"]
    tests = list(grading.get("fail_to_pass") or []) + list(grading.get("pass_to_pass") or [])
    if not tests:
        return GradeResult(passed=False, setup_failed=True, detail="Manifest has no fail_to_pass/pass_to_pass tests")

    try:
        venv_python = _isolated_venv_python(workspace_dir)
    except Exception as exc:
        return GradeResult(passed=False, setup_failed=True, detail=f"Failed to prepare isolated venv: {exc}")

    # Best-effort generic install -- the whole reason the official SWE-bench
    # harness uses a per-instance Docker image is that this step is NOT
    # generically reliable across instances/Python versions. A failure
    # here is a setup_failed, not a graded loss. pytest is installed
    # alongside the target package since this venv starts out bare (unlike
    # sys.executable's venv, which happened to already have one -- that
    # coincidence is exactly what masked the isolation bug above).
    install_proc = _run(
        [venv_python, "-m", "pip", "install", "-e", ".", "pytest", "-q"], cwd=workspace_dir, timeout=900,
    )
    if install_proc.returncode != 0:
        return GradeResult(
            passed=False, setup_failed=True,
            detail=_truncate("pip install -e . failed:\n" + install_proc.stdout + install_proc.stderr),
        )

    test_proc = _run([venv_python, "-m", "pytest", *tests, "-q"], cwd=workspace_dir, timeout=300)
    output = test_proc.stdout + test_proc.stderr

    # Exit code alone conflates two very different situations: pytest ran
    # the requested tests and some failed (a real graded loss) vs. pytest
    # never got that far at all -- an import crash during collection (an
    # incompatible transitive dependency version, common on a generic `pip
    # install -e .` with no per-instance environment pinning) or the named
    # test simply not existing in this checkout. Both return nonzero, but
    # only the first is actually about the agent's fix. Verified live: a
    # 30-task run had multiple instances (flask ImportError on werkzeug,
    # xarray/django "no tests ran"/"found no collectors") that never
    # produced a pytest summary line at all, yet were being reported as a
    # graded FAIL -- same class of bug _classify_bugsinpy_output() exists
    # to avoid for BugsInPy, just not applied here yet.
    if _PYTEST_SUMMARY_RE.search(output):
        return GradeResult(passed=test_proc.returncode == 0, detail=_truncate(output))

    return GradeResult(
        passed=False, setup_failed=True,
        detail=_truncate("pytest produced no summary line (crashed before running any test):\n" + output),
    )


# ── dispatch ─────────────────────────────────────────────────────────────

_PREPARE = {
    "quixbugs_isolated": _prepare_quixbugs,
    "bugsinpy_checkout": _prepare_bugsinpy,
    "git_base_commit": _prepare_swebench,
}
_GRADE = {
    "quixbugs_isolated": _grade_quixbugs,
    "bugsinpy_checkout": _grade_bugsinpy,
    "git_base_commit": _grade_swebench,
}


def prepare_workspace(instance: Dict[str, Any], dest: Path) -> Path:
    strategy = instance["workspace"]["strategy"]
    fn = _PREPARE.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown workspace strategy: {strategy!r}")
    # dest itself is deliberately NOT pre-created here: shutil.copytree
    # (quixbugs) requires its destination to not already exist, while
    # bugsinpy-checkout and git clone (bugsinpy/swebench) create it
    # themselves. Only guarantee the parent exists.
    dest.parent.mkdir(parents=True, exist_ok=True)
    return fn(instance, dest)


def grade(instance: Dict[str, Any], workspace_dir: Path) -> GradeResult:
    strategy = instance["workspace"]["strategy"]
    fn = _GRADE.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown workspace strategy: {strategy!r}")
    return fn(instance, workspace_dir)
