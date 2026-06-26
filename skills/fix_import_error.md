---
name: fix_import_error
trigger_keywords: [ModuleNotFoundError, ImportError, cannot import, no module named]
summary: Fix a broken Python import by locating the correct module path, checking __init__.py files, and updating the import statement.
---

## Procedure
1. Run `identify_error` on the traceback to extract the missing module name.
2. Use `search_code` with the missing module name to find the actual file.
3. Check whether `__init__.py` exports it.
4. Use `replace_in_file` to fix the import statement.
5. Run `run_tests` to validate.