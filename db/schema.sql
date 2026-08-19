-- Persistence schema for the Code Error Agent.
--
-- Maps directly onto what api/server.py currently keeps in the in-memory
-- `_sessions` / `_runs` dicts:
--   _sessions[session_id]                 -> sessions row
--   _sessions[session_id]["history"]      -> messages rows (session_id FK)
--   _runs[run_id]                          -> runs row
--
-- Not covered here on purpose: AgentMemory's long-term insights
-- (.agent_memory/<namespace>/memory.jsonl). That's a separate, smaller
-- migration tied to the RAG-unification design discussion, not part of the
-- api/server.py persistence layer -- see the "database retrieve" direction
-- notes for why it may eventually want its own table + embedding column
-- (pgvector) rather than living here.
--
-- See cleanup.sql for the retention/cleanup procedures that keep these
-- tables from growing forever.

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    repo_label   TEXT,
    repo_root    TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active  TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS messages (
    id           BIGSERIAL PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user', 'agent')),
    content      TEXT NOT NULL,
    run_id       TEXT,             -- NULL for light-chat replies; set for full agent runs
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages (session_id, created_at);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
    task            TEXT NOT NULL,
    repo_url        TEXT,
    run_status      TEXT,           -- pending | running | completed | failed | skipped
    validation      TEXT,           -- not_run | passed | failed | error
    stop_reason     TEXT,
    replan_count    INT,
    tool_call_count INT,
    files_modified  JSONB,
    final_answer    TEXT,
    result          JSONB,          -- full result blob (plan, per-step status, etc.)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_runs_session_created
    ON runs (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_runs_status
    ON runs (run_status);

-- Retention-friendly indexes: cleanup.sql scans by these columns to find
-- stale rows without a sequential scan. Without them, a nightly cleanup job
-- gets slower (and locks rows for longer) as the tables grow -- exactly
-- backwards from what a cleanup job should do.
CREATE INDEX IF NOT EXISTS idx_sessions_last_active
    ON sessions (last_active) WHERE closed_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_runs_finished_at
    ON runs (finished_at) WHERE finished_at IS NOT NULL;
