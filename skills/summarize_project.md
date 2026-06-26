---
name: summarize_project
trigger_keywords: [what does, what is, describe the project, project overview, project summary, what can, how does, purpose of]
summary: Answer "what does this project do" by reading documentation and entry points first, never by guessing.
---

## Procedure

1. **list_files** at depth 1. Identify: README*, pyproject.toml, setup.py, setup.cfg, Cargo.toml, package.json, go.mod, pom.xml, Makefile.
2. **read_file README** (try README.md, README.rst, README.txt, README in that order). This is the primary source of truth. Stop here if the answer is in the README.
3. **read_file pyproject.toml or setup.py** to get the package name, description field, and entry_points / console_scripts.
4. **list_files** the main package directory (usually the directory matching the package name) at depth 1 to see its modules.
5. **read_file** the main entry point (e.g. `__main__.py`, `cli.py`, `main.py`, `app.py`) — first 60 lines only.
6. Synthesize only from what you actually read. If a file was not found or unreadable, say so. Do not infer purpose from directory names alone.
