---
name: upgrade_dependency
trigger_keywords: [upgrade, bump version, update dependency, update dependencies, migrate to]
summary: Upgrade a dependency by checking every call site for breaking-change risk before bumping the version, then validate with tests.
---

## Procedure
1. Use `list_files`/`search_code` to locate every manifest that pins the dependency (requirements.txt, package.json, pyproject.toml -- a version can be pinned in more than one place).
2. Use `search_code`/`retrieve_context` to find every call site using the dependency's API -- an upgrade can silently break a call site using a removed/changed API.
3. Use `replace_in_file` to update the version pin(s) -- a small, targeted edit, not a manifest rewrite.
4. Run `run_tests`/`run_command` to catch breaking changes; a version bump is only safe once tests actually pass against it.
