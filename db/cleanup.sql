-- Retention / cleanup for the persistence layer (see schema.sql).
--
-- Three things this handles:
--   1. Close idle sessions that were never explicitly DELETE'd (mirrors the
--      30-minute in-memory watchdog in api/server.py -- once the app reads
--      from Postgres instead of the _sessions dict, Redis TTL should own
--      this instead; this stays as a correctness backstop either way).
--   2. Batched delete of sessions closed longer than the retention window.
--      messages/runs referencing them cascade automatically (see schema.sql's
--      ON DELETE CASCADE / ON DELETE SET NULL).
--   3. Batched delete of old *unowned* finished runs -- one-shot /api/run
--      runs have no session_id, so step 2's cascade never reaches them.
--
-- Batched, not a single DELETE: a single `DELETE FROM runs WHERE ...` that
-- matches hundreds of thousands of rows holds a lock on the table for the
-- whole statement. These procedures delete LIMIT-bounded batches and COMMIT
-- between batches (COMMIT inside a loop requires a PROCEDURE + CALL --
-- plain functions/anonymous DO blocks can't do that as of Postgres 11+).
--
-- Usage:
--   psql "$DATABASE_URL" -f db/cleanup.sql          -- (re)creates the procedures
--   psql "$DATABASE_URL" -c "CALL cleanup_old_sessions(); CALL cleanup_orphan_runs();"
--
-- Scheduling options (pick one -- see the trade-off notes in the write-up):
--   a) App-level periodic task (recommended to start): a background job
--      calling the two CALL statements above once a day. If/when the task
--      queue (RQ) is in place, this is a natural periodic RQ job -- reuses
--      infra you already have instead of adding a new moving part.
--   b) pg_cron extension: `SELECT cron.schedule('cleanup', '0 3 * * *',
--      'CALL cleanup_old_sessions(); CALL cleanup_orphan_runs();');` --
--      keeps scheduling inside Postgres, but not every managed/hosted
--      Postgres allows installing extensions.
--   c) External cron (system crontab / k8s CronJob) running the psql command
--      above -- fully decoupled, but one more component to deploy/monitor.

CREATE OR REPLACE PROCEDURE close_idle_sessions(idle_minutes INT DEFAULT 30)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE sessions
    SET closed_at = now()
    WHERE closed_at IS NULL
      AND last_active < now() - (idle_minutes || ' minutes')::interval;
END;
$$;

CREATE OR REPLACE PROCEDURE cleanup_old_sessions(retention_days INT DEFAULT 30, batch_size INT DEFAULT 1000)
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count INT;
BEGIN
    LOOP
        DELETE FROM sessions
        WHERE session_id IN (
            SELECT session_id FROM sessions
            WHERE closed_at IS NOT NULL
              AND closed_at < now() - (retention_days || ' days')::interval
            LIMIT batch_size
        );
        GET DIAGNOSTICS deleted_count = ROW_COUNT;
        COMMIT;  -- release the batch's locks before starting the next one
        EXIT WHEN deleted_count = 0;
    END LOOP;
END;
$$;

CREATE OR REPLACE PROCEDURE cleanup_orphan_runs(retention_days INT DEFAULT 30, batch_size INT DEFAULT 1000)
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count INT;
BEGIN
    LOOP
        DELETE FROM runs
        WHERE run_id IN (
            SELECT run_id FROM runs
            WHERE session_id IS NULL           -- one-shot /api/run runs only;
              AND finished_at IS NOT NULL       -- session-owned runs are handled
              AND finished_at < now() - (retention_days || ' days')::interval  -- by cleanup_old_sessions' cascade
            LIMIT batch_size
        );
        GET DIAGNOSTICS deleted_count = ROW_COUNT;
        COMMIT;
        EXIT WHEN deleted_count = 0;
    END LOOP;
END;
$$;

-- Run once now with the defaults (30 min idle, 30 day retention). Adjust the
-- arguments per call if you want different windows for a given run, e.g.
-- CALL cleanup_old_sessions(retention_days => 7);
CALL close_idle_sessions();
CALL cleanup_old_sessions();
CALL cleanup_orphan_runs();
