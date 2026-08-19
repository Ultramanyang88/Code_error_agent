"""
Structured (JSON-lines) logging for the agent's operational events --
separate from the human-facing `print()` narration in core/executor.py,
main.py, etc. (that's CLI UX, left alone). Every call site logs through
`log_event()` with a stable set of fields, so swapping the output sink
(file -> stdout -> OTel exporter) is a change to this module only.

Env vars: AGENT_LOG_DIR (default "logs"), AGENT_LOG_LEVEL (default "INFO").
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Any, Optional

_LOGGER_NAME = "agent.events"
_configured = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _configure() -> logging.Logger:
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)

    if _configured:
        return logger

    logger.setLevel(os.environ.get("AGENT_LOG_LEVEL", "INFO"))
    logger.propagate = False  # don't also go through root -> stderr as plain text

    log_dir = Path(os.environ.get("AGENT_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "agent-events.jsonl",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    _configured = True
    return logger


def log_event(event: str, **fields: Any) -> None:
    """
    Emit one structured JSON-lines event.

    Example:
        log_event("tool_call", run_id=state.run_id, tool="read_file",
                   success=True, duration_ms=12.4)
    """
    logger = _configure()
    logger.info(event, extra={"fields": fields})


class timed:
    """
    Context manager that measures wall-clock duration in milliseconds.

    Usage:
        with timed() as t:
            do_work()
        log_event("thing_done", duration_ms=t.duration_ms)
    """

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        self.duration_ms: Optional[float] = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.duration_ms = round((time.perf_counter() - self._start) * 1000, 2)
