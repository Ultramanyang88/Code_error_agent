# Evaluation Sets

This directory stores small, versioned manifests for external evaluation suites.
Large cloned repositories and generated workspaces stay local under `cache/` and
`workspaces/`; both directories are ignored by Git.

## Build The Portfolio Set

Install the optional dataset dependency in the project virtual environment:

```bash
source .venv/bin/activate
pip install -r testcase/requirements-eval.txt
```

Build the default 30-task set:

```bash
python testcase/build_eval_set.py
```

The command selects:

- 10 small Python repair tasks from QuixBugs.
- 10 real Python bugs from BugsInPy.
- 10 repository-level issues from SWE-bench Lite.

The generated manifest is written to
`testcase/eval_sets/manifests/portfolio_30.json`. Selection is deterministic for
the same source revisions and `--seed` value. It balances repositories, inferred
error categories, and patch difficulty where the source data permits it.

Useful options:

```bash
python testcase/build_eval_set.py --help
python testcase/build_eval_set.py --seed 7
python testcase/build_eval_set.py --quixbugs-count 5 --bugsinpy-count 5 --swebench-count 20
python testcase/build_eval_set.py --refresh
```

`--refresh` updates cached Git metadata repositories before rebuilding the
manifest. SWE-bench data is cached by Hugging Face automatically.

## Running The Set

`testcase/build_eval_set.py` only produces the catalog above -- it doesn't
materialize workspaces or run the agent. That's `testcase/run_portfolio_eval.py`
(the workspace-prep + grading logic lives in `eval_sets/workspace.py`):

```bash
# Fallback mode (no LLM), QuixBugs only -- fastest way to sanity-check the wiring
python testcase/run_portfolio_eval.py --source quixbugs

# LLM mode, everything
python testcase/run_portfolio_eval.py --llm --provider openai_compatible --model qwen2.5-coder:7b

# Single instance, keep its workspace around afterward for inspection
python testcase/run_portfolio_eval.py --instance quixbugs__max_sublist_sum --keep-workspaces
```

Results (including a per-instance `trace.jsonl`/`result.json`, same shape as
`testcase/run_eval.py`'s) land under `testcase/results/portfolio/`. This is a
**separate** runner/results tree from `testcase/run_eval.py`'s original 3-task
fixture set on purpose -- that one stays a fast, dependency-free sanity check
of the agent loop itself; this one hits real network and per-source setup
that can legitimately fail on its own (a broken pip install shouldn't be able
to fail the quick check).

Each source's grading result can be `passed`, `failed`, or `setup_failed` --
the third bucket means the environment never got far enough to run the
agent's fix at all (bad checkout, failed compile/install), and is excluded
from the pass rate rather than counted as a loss. Concretely, per source:

- **quixbugs**: isolated workspace = `python_programs/` (buggy code, what
  the agent edits) + `python_testcases/` (tests and their shared helpers,
  e.g. `load_testdata.py`) + `json_testcases/` (test data) + root
  `conftest.py` (defines the `pytest.use_correct` toggle the tests import
  against). `correct_python_programs/`/`correct_java_programs/` (the
  reference fixes) are never copied in -- not excluded after the fact, just
  never touched. Graded with `pytest <test_file>` (no `--correct`, so it
  exercises whatever the agent left in `python_programs/`).
- **bugsinpy**: drives the real BugsInPy bash framework
  (`bugsinpy-checkout -v 0` for the buggy version, then `bugsinpy-compile` +
  `bugsinpy-test -r`) against the actual historical project repo. Note the
  framework clones one level down (`<workspace>/<project>/`, not
  `<workspace>/` itself) -- `prepare_workspace()`'s return value is the real
  path to use, not the directory you passed in. `bugsinpy-compile` builds a
  fresh venv and installs the bug's pinned `requirements.txt`, which can be
  slow and can fail outright on old pins that no longer build on a modern
  Python/OS -- a `setup_failed`, not a graded loss. `bugsinpy-test`'s own
  script doesn't propagate a clean exit code, so grading parses its captured
  pytest/unittest output for pass/fail markers instead
  (`workspace._bugsinpy_output_indicates_pass`).
- **swebench**: **simplified grading, not the official SWE-bench Docker
  harness.** Clones the real repo at `base_commit` and runs a generic
  `pip install -e .` in a fresh venv built just for that one workspace
  (`.agent_eval_venv/`, **never** this project's own interpreter --
  `pip install -e .` against `sys.executable` was tried first and, verified
  live, installed the target repo's own package as an editable link
  straight into this project's `.venv`; grading `pytest-dev/pytest`
  silently replaced this project's real `pytest` with one pointed at a
  workspace directory that gets deleted after the run, breaking
  `pytest testcase/test_all.py` itself until it was manually repaired --
  several other instances, django/flask/sphinx/sympy/xarray, did the same
  to their own package names), then the instance's `fail_to_pass`/
  `pass_to_pass` test lists directly. The official harness exists
  specifically because that install step isn't generically reliable
  across instances/Python versions (verified live: `psf__requests-1963`
  fails to build with a modern pip against its old `setup.py`) -- expect a
  meaningful fraction of instances to land in `setup_failed` rather than a
  real pass/fail. This tradeoff (lighter weight, lower fidelity, no
  per-instance Docker image) was chosen deliberately; swapping in the
  official harness later is a bigger, separate piece of work.

## Manifest Safety

The manifest intentionally excludes reference fixes (`patch` and `test_patch`).
Graders should load those fields from the original dataset by `instance_id` only
after the agent has produced its own patch. This prevents the agent from seeing
the answer during inference.
