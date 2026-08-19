# Code Error Agent

Code Error Agent is an autonomous coding agent for working with real repositories. It can inspect code, plan a task, call tools, edit files, run tests, and revise its plan when validation fails.

It supports more than bug fixing: project analysis, feature implementation, refactoring, test writing, dependency upgrades, and code review. You can use it from the CLI or through the FastAPI web UI.

## Summary

The agent follows a simple loop:

1. Classify the user request and create a task-specific plan.
2. Retrieve relevant project context with file search and RAG.
3. Execute each step with built-in tools or MCP tools.
4. Validate changes with tests or command output.
5. Replan when validation fails, up to the configured budget.

Main features:

- CLI runner in `main.py`.
- FastAPI backend and browser chat UI in `api/`.
- Built-in tools for reading files, searching code, applying patches, running commands/tests, and checking diffs.
- RAG layer using FAISS and sentence-transformers.
- Skill playbooks in `skills/` for common coding tasks.
- MCP support for both exposing this project as a tool server and consuming external MCP servers.
- Optional Postgres, Redis/RQ, and Docker sandbox support.

## Architecture

```text
main.py
  CLI entry point. Creates the agent state, loads tools/MCP servers, and runs the planner-executor loop.

core/
  state.py           Agent state, plan steps, tool results, budgets, run status
  planner.py         LLM planner plus rule-based fallback planner
  executor.py        Executes plan steps through tools and builds the final answer
  memory.py          Session and long-term agent memory
  logging_setup.py   Structured JSONL logging

tools/
  tools.py           Built-in tool implementations
  specs.py           Tool schemas and categories
  registry.py        Combines built-in and MCP tool metadata for the planner

rag/
  indexer.py         Code chunking and FAISS index building
  retrieve.py        Hybrid vector/keyword retrieval and reranking
  federated.py       Combines code retrieval with agent memory

skills/
  *.md               Task playbooks for bugs, features, refactors, tests, upgrades, etc.
  registry.py        Loads skills and matches them by trigger keywords

agent_mcp/
  server.py          Exposes built-in tools/resources/prompts as an MCP server
  client.py          Connects to external MCP servers
  config.py          Loads mcp_servers.json or MCP_SERVERS_CONFIG

api/
  server.py          FastAPI app, sessions, runs, SSE streaming, history endpoints
  static/index.html  Browser chat UI

db/
  schema.sql         Postgres schema
  store.py           Optional Postgres persistence
  cache.py           Optional Redis TTL/pub-sub support
  queue.py           Optional RQ task queue

sandbox/
  docker_sandbox.py  Optional Docker execution sandbox
  Dockerfile         Sandbox image

testcase/
  test_*.py          Tests and integration checks
  run_eval.py        Evaluation runner
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create local environment config:

```bash
cp .env.example .env
```

Then edit `.env` as needed. For OpenAI-compatible models, set:

```env
OPENAI_API_KEY=sk-...
```

## Run From CLI

Fallback mode, without an LLM:

```bash
python main.py
```

OpenAI-compatible mode:

```bash
python main.py --llm \
  --provider openai_compatible \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --task "Analyze this repository and summarize its architecture"
```

Ollama mode:

```bash
ollama pull qwen2.5-coder:7b
python main.py --llm --provider ollama --model qwen2.5-coder:7b
```

## Run The Web UI

Start the API server:

```bash
python api/server.py
```

Open:

```text
http://localhost:8080
```

The UI lets you start a session from a GitHub URL or uploaded `.py`/`.zip` file, chat with the agent, and stream execution logs.

## Run The Full Local Stack

This starts Postgres, Redis, the optional Docker sandbox, an RQ worker, and the API server.

Requirements:

- Docker running
- Python virtual environment created
- Dependencies installed
- `.env` configured if you want LLM access

Run:

```bash
./scripts/dev-up.sh
```

Useful options:

```bash
./scripts/dev-up.sh --no-queue
./scripts/dev-up.sh --no-sandbox
```

Stop the API server and worker with `Ctrl+C`. Stop Postgres/Redis separately:

```bash
docker compose down
```

## Optional Services

Postgres and Redis:

```bash
docker compose up -d postgres redis
export DATABASE_URL="postgresql://agent:agent_dev_password@localhost:5432/agent"
export REDIS_URL="redis://localhost:6379/0"
python api/server.py
```

RQ worker:

```bash
export REDIS_URL="redis://localhost:6379/0"
export AGENT_QUEUE_ENABLED=1
rq worker agent-runs --worker-class rq.worker.SimpleWorker
```

Docker sandbox:

```bash
docker build -t agent-sandbox:latest sandbox/
export AGENT_SANDBOX_ENABLED=1
python api/server.py
```

## Tests

```bash
python -m pytest testcase/test_all.py -v
python testcase/test_tools.py
python testcase/test_rag.py
```

Optional integration tests:

```bash
DATABASE_URL="postgresql://agent:agent_dev_password@localhost:5432/agent" \
  python -m pytest testcase/test_db_store.py -v

python -m pytest testcase/test_sandbox.py -v

REDIS_URL="redis://localhost:6379/0" \
  python -m pytest testcase/test_cache_queue.py -v
```

## Configuration

Common environment variables:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | API key for OpenAI-compatible providers |
| `DATABASE_URL` | Optional Postgres DSN |
| `REDIS_URL` | Optional Redis URL |
| `AGENT_QUEUE_ENABLED` | Enable RQ when set to `1`, `true`, or `yes` |
| `AGENT_SANDBOX_ENABLED` | Enable Docker sandbox when set to `1`, `true`, or `yes` |
| `MCP_SERVERS_CONFIG` | Path to a custom MCP server config file |
| `AGENT_LOG_DIR` | Directory for JSONL logs |
| `AGENT_LOG_LEVEL` | Log level |

See `mcp_servers.json.example` for MCP configuration examples.
