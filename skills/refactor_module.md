---
name: refactor_module
trigger_keywords: [refactor, clean up, simplify, restructure, reorganize, extract method, dedupe]
summary: Refactor code while preserving external behavior -- find every call site first, capture a test baseline, then change, then re-verify against that baseline.
---

## Procedure
1. Use `search_code`/`retrieve_context` to find every call site the refactor will touch -- a refactor that misses one breaks the build.
2. Use `read_file` on the current implementation and every call site found above.
3. Run `run_tests` to capture a pre-refactor baseline (what already passes/fails).
4. Use `apply_patch`/`replace_in_file` to apply the refactor across all identified call sites in one pass.
5. Run `run_tests` again and compare against the baseline from step 3 -- the refactor only succeeded if the result is the same (or strictly better), never a new failure.
