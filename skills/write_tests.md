---
name: write_tests
trigger_keywords: [write test, add test, add unit test, write unit test, test coverage, add coverage]
summary: Write new test cases that follow the project's existing test conventions and are grounded in the actual behavior of the code under test.
---

## Procedure
1. Use `list_files`/`search_code` to find the existing test suite's location and conventions (framework, naming, fixture style).
2. Use `read_file`/`retrieve_context` on the code that needs coverage -- understand its actual behavior and edge cases, not assumed behavior.
3. Use `write_file`/`replace_in_file` to add the test cases (happy path + edge cases), matching the existing test file's style.
4. Run `run_tests` to confirm the new tests pass and nothing existing broke.
