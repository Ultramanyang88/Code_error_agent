# Code Error Agent

An autonomous coding agent that finds and fixes bugs in Python repositories. Submit a GitHub URL or file, chat with the agent, and get grounded answers backed by the actual code — not hallucinations.

## How it works

The agent runs a loop:

1. **Planner** — sees the repo file listing before planning, generates a task-specific sequence of steps
2. **Executor** — runs each step via tool calls; logs each tool invocation and result to the terminal
3. **Skill registry** — dispatches known task patterns (e.g. "what does this project do") to hand-written playbooks before falling back to the LLM planner
4. **Validator** — checks test results; if tests fail the planner replans (capped at `max_replans`)

RAG layer: FAISS + sentence-transformers for hybrid (vector + keyword) recall, cross-encoder reranking, relevance floor (results below score −4.0 are discarded). The knowledge base includes synthetic chunks for tool specs and skill playbooks so the planner can look up how to use tools. README and documentation files get boosted scoring for overview queries.

MCP server exposes all built-in tools over stdio for external orchestration.

## Project layout

```
core/
  state.py      — AgentState, AgentBudget, PlanStep, ToolResult
  planner.py    — LLM planner (with repo listing in prompt) + deterministic fallback
  executor.py   — step runner, tool dispatch, schema validation, grounded final summary
  memory.py     — short-term (recent results) + long-term (JSONL insights)

tools/
  tools.py      — list_files, read_file, search_code, write_file,
                  replace_in_file, apply_patch, run_command, run_tests,
                  identify_error, git_diff, retrieve_context
  specs.py      — JSON schema definitions for all tools

rag/
  indexer.py    — AST-based chunker + FAISS index builder (injects tool + skill chunks)
  retrieve.py   — hybrid vector + keyword search, cross-encoder rerank, relevance floor
  embedder.py   — sentence-transformers wrapper

skills/
  registry.py           — loads .md skill files, keyword-based dispatch
  fix_import_error.md
  debug_test_failure.md
  summarize_project.md  — read README → pyproject.toml → entry point, in order

agent_mcp/
  server.py     — MCP server exposing all built-in tools over stdio
  client.py     — MCP client wrapping external tool servers

api/
  server.py     — FastAPI backend; session-based multi-turn chat
  static/
    index.html  — Chat UI (session per repo, conversation thread, live log stream)

testcase/
  test_all.py   — Full test suite (106 tests, no server required)
  run_eval.py   — Eval harness over task_00x_* bug scenarios
  tasks/        — Three eval tasks: division-by-zero, operator bug, import error
  judge.py      — LLM-as-judge scoring via gpt-4o-mini

main.py         — CLI entry point
llm.py          — LLM client (OpenAI-compatible or Ollama, reads OPENAI_API_KEY)
```

## Quickstart

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run in fallback mode (no LLM — rule-based planner)
python main.py

# 3. Run with OpenAI
OPENAI_API_KEY=sk-... python main.py --llm --provider openai_compatible \
  --base-url https://api.openai.com/v1 --model gpt-4o-mini

# 4. Run with a local LLM (Ollama)
ollama pull qwen2.5-coder:7b
python main.py --llm --provider ollama --model qwen2.5-coder:7b

# 5. Start the web UI (chat interface)
OPENAI_API_KEY=sk-... python api/server.py
open http://localhost:8080
```

## Chat UI

The web interface is session-based — the repo is cloned once per session, and you can send follow-up messages without re-cloning.

Flow:
1. Paste a GitHub URL (or upload a `.py`/`.zip` file) and optionally write a first message
2. Configure LLM provider in the accordion (or leave as "Fallback")
3. Click **Start session** — the agent runs and streams progress to the log panel
4. Each agent response appears as a result card in the conversation thread
5. Type follow-up questions in the input box at the bottom
6. Click **New session** in the header to start over with a different repo

## Tests

```bash
# Full unit test suite (106 tests, no server or LLM needed)
python -m pytest testcase/test_all.py -v

# Tool smoke test
python testcase/test_tools.py

# RAG smoke test
python testcase/test_rag.py
```

## API

Server runs at `http://localhost:8080`.

### Session endpoints (multi-turn)

```bash
# Create a session (clone repo once)
curl -X POST http://localhost:8080/api/session \
  -F "repo_url=https://github.com/you/repo"

# → {"session_id": "a1b2c3d4", "repo_label": "https://..."}

# Send a message in the session
curl -X POST http://localhost:8080/api/session/a1b2c3d4/message \
  -F "task=What does this project do?" \
  -F "llm_provider=openai_compatible" \
  -F "llm_base_url=https://api.openai.com/v1" \
  -F "llm_model=gpt-4o-mini"

# → {"run_id": "b5c6d7e8"}

# Close a session and clean up workspace
curl -X DELETE http://localhost:8080/api/session/a1b2c3d4
```

### Run endpoints (one-shot)

```bash
# Start a one-shot run (no session, workspace cleaned up after 5 min)
curl -X POST http://localhost:8080/api/run \
  -F "repo_url=https://github.com/you/repo" \
  -F "task=Fix the divide-by-zero bug"

# Stream live log (SSE)
curl -N http://localhost:8080/api/run/b5c6d7e8/stream

# Poll run result
curl http://localhost:8080/api/run/b5c6d7e8
```

### History / delete (PostgreSQL — not active by default)

```bash
curl http://localhost:8080/api/history?limit=20
curl -X DELETE http://localhost:8080/api/run/b5c6d7e8
```

## Evaluation

```bash
# Run all three eval tasks
python testcase/run_eval.py

# With LLM judge (gpt-4o-mini scoring)
OPENAI_API_KEY=sk-... python testcase/run_eval.py --judge

# Save / compare regression baseline
python testcase/run_eval.py --save-baseline
python testcase/run_eval.py --compare-baseline
```

## PostgreSQL (optional)

By default, runs are stored in memory and sessions are cleaned up after 30 min of inactivity. To persist history:

1. Create the table:
```sql
CREATE TABLE runs (
    run_id      TEXT PRIMARY KEY,
    task        TEXT,
    repo_url    TEXT,
    status      TEXT,
    result      JSONB,
    created_at  TIMESTAMPTZ DEFAULT now(),
    finished_at TIMESTAMPTZ
);
```

2. Install the driver and set the DSN:
```bash
pip install asyncpg
export DATABASE_URL="postgresql://user:pass@localhost:5432/agent"
```

3. Uncomment the `# ── [DB PLACEHOLDER]` block in [api/server.py](api/server.py).

## Adding a skill

Create `skills/your_skill.md`:

```markdown
---
name: your_skill
trigger_keywords: [keyword1, keyword2]
summary: One sentence describing what this skill does.
---

## Procedure
1. Step one (include which tool to use).
2. Step two.
```

Skills are picked up automatically. Trigger keywords are matched against the current step task and recent error messages before the LLM planner is called.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--llm` | off | Enable LLM mode |
| `--provider` | `openai_compatible` | `openai_compatible` or `ollama` |
| `--base-url` | `http://localhost:8000` | LLM API base URL |
| `--model` | `qwen2.5-coder:7b` | Model name |
| `--task` | repo analysis | Task description |

`OPENAI_API_KEY` is read automatically from the environment when using the `openai_compatible` provider.

Budget defaults (`core/state.py`): 8 plan steps, 30 tool calls, 6 replans, 600 s deadline.
