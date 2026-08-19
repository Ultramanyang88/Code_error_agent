---
name: implement_feature
trigger_keywords: [implement a, add a feature, new feature, add support for, create a new, build a]
summary: Implement a new feature by following the codebase's existing patterns rather than inventing a new structure, then validate it with tests.
---

## Procedure
1. Use `list_files`/`retrieve_context` to find where similar functionality already lives.
2. Use `search_code` to find an existing feature to model the new one on -- prefer extending an established pattern over introducing a new one.
3. Use `read_file` on the integration points (where the new feature needs to plug in).
4. Use `write_file`/`apply_patch`/`replace_in_file` to implement the feature.
5. Run `run_tests`; if the feature has no coverage yet, add tests for it rather than shipping it unverified.
