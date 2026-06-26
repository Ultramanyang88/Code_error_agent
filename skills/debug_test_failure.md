---
name: debug_test_failure
trigger_keywords: [FAILED, AssertionError, pytest, test failed, run_tests]
summary: Debug a failing test by reading the test file, identifying the assertion, tracing back to the source function, and applying a minimal fix.
---

## Procedure
1. Run `identify_error` on the pytest output.
2. Use `read_file` on the failing test file.
3. Use `search_code` to find the function under test.
4. Use `read_file` on the source function.
5. Apply a minimal fix with `replace_in_file`.
6. Re-run `run_tests`.