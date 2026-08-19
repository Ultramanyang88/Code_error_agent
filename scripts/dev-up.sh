#!/usr/bin/env bash
# Starts the whole local dev stack: Postgres + Redis (docker compose), the
# sandbox image (if missing), an RQ worker, and the API server -- then opens
# the chat UI in your browser.
#
# Usage:
#   ./scripts/dev-up.sh              # full stack: PG + Redis + sandbox + RQ
#   ./scripts/dev-up.sh --no-queue   # skip the RQ worker (runs jobs in-process instead)
#   ./scripts/dev-up.sh --no-sandbox # skip building/using the sandbox image
#
# Ctrl+C stops the server + worker (docker compose is left running -- run
# `docker compose down` yourself when you're done with it for the session).
set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env if present (see .env.example) and export everything in it to
# child processes -- covers the `rq worker` CLI below directly, which needs
# $REDIS_URL at connection time before it has imported any Python module
# that could load .env itself. .env values win over anything already
# exported in this shell; unset a var here (or don't put it in .env) if you
# want an inline `VAR=... ./scripts/dev-up.sh` override to take effect instead.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

USE_QUEUE=1
USE_SANDBOX=1
for arg in "$@"; do
  case "$arg" in
    --no-queue) USE_QUEUE=0 ;;
    --no-sandbox) USE_SANDBOX=0 ;;
  esac
done

if ! docker info >/dev/null 2>&1; then
  echo "[!] Docker daemon isn't running. Start Docker Desktop (or 'colima start'), then re-run this script."
  exit 1
fi

echo "[*] Starting Postgres + Redis..."
docker compose up -d postgres redis

export DATABASE_URL="${DATABASE_URL:-postgresql://agent:agent_dev_password@localhost:5432/agent}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"

if [ "$USE_SANDBOX" = "1" ]; then
  if ! docker image inspect agent-sandbox:latest >/dev/null 2>&1; then
    echo "[*] Building sandbox image (first run only, ~30-60s)..."
    docker build -t agent-sandbox:latest sandbox/
  fi
  export AGENT_SANDBOX_ENABLED=1
  echo "[*] Sandbox: enabled (run_command/run_tests/apply_patch/git-clone run in Docker)"
else
  echo "[*] Sandbox: disabled (--no-sandbox)"
fi

WORKER_PID=""
cleanup() {
  echo ""
  echo "[*] Shutting down..."
  [ -n "$WORKER_PID" ] && kill "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT

if [ "$USE_QUEUE" = "1" ]; then
  export AGENT_QUEUE_ENABLED=1
  # --worker-class SimpleWorker: RQ's default worker forks a subprocess per
  # job, which crashes on macOS once PyTorch/sentence-transformers (the RAG
  # embedder) has touched Metal/GPU state -- Apple's Objective-C runtime
  # treats that as unsafe across fork() and aborts the process. SimpleWorker
  # runs jobs in the worker's own process instead (no fork), which sidesteps
  # the crash entirely. On Linux this crash doesn't occur, but SimpleWorker
  # still works fine there -- no reason to special-case the platform.
  echo "[*] Starting RQ worker (SimpleWorker -- see comment in this script for why)..."
  OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
    .venv/bin/rq worker agent-runs --worker-class rq.worker.SimpleWorker &
  WORKER_PID=$!
  echo "[*] Task queue: enabled (worker PID $WORKER_PID)"
else
  echo "[*] Task queue: disabled (--no-queue) -- runs execute in-process"
fi

echo "[*] Starting API server on http://localhost:8080 ..."
( sleep 1.5 && command -v open >/dev/null && open http://localhost:8080 ) &
.venv/bin/python api/server.py
