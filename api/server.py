"""
FastAPI backend for the Code Error Agent web UI.

Run:
    python api/server.py
    # or
    uvicorn api.server:app --reload --port 8080
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="Code Error Agent")

# Serve static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory run registry: run_id → {queue, done, result}
_runs: Dict[str, Dict[str, Any]] = {}

# Session registry: session_id → {tmp_dir, repo_root, repo_label, history, last_active}
_sessions: Dict[str, Dict[str, Any]] = {}


def _build_contextual_task(message: str, history: List[Dict[str, str]]) -> str:
    """Prepend recent conversation history so the agent has multi-turn context."""
    if not history:
        return message
    lines = []
    for m in history[-8:]:  # last 4 exchanges
        role = "User" if m["role"] == "user" else "Agent"
        lines.append(f"[{role}]: {m['content']}")
    return "Conversation history:\n" + "\n".join(lines) + f"\n\n[User]: {message}"

# ── [DB PLACEHOLDER] ──────────────────────────────────────────────────────────
# PostgreSQL connection pool.
# Uncomment and configure when ready to persist run history.
#
# import asyncpg
# DB_DSN = os.environ.get("DATABASE_URL", "postgresql://user:pass@localhost:5432/agent")
# _db_pool: Optional[asyncpg.Pool] = None
#
# @app.on_event("startup")
# async def startup():
#     global _db_pool
#     _db_pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=10)
#
# @app.on_event("shutdown")
# async def shutdown():
#     if _db_pool:
#         await _db_pool.close()
#
# Schema (run once):
#   CREATE TABLE runs (
#       run_id      TEXT PRIMARY KEY,
#       task        TEXT,
#       repo_url    TEXT,
#       status      TEXT,
#       result      JSONB,
#       created_at  TIMESTAMPTZ DEFAULT now(),
#       finished_at TIMESTAMPTZ
#   );
# ─────────────────────────────────────────────────────────────────────────────


# ── workspace helpers ─────────────────────────────────────────────────────────

def _clone_repo(url: str, dest: Path) -> None:
    url = url.strip()
    if not url.startswith("http"):
        raise ValueError(f"Only HTTPS URLs are supported: {url}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed:\n{result.stderr[:800]}")


def _write_uploaded_file(content: bytes, filename: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if filename.endswith(".zip"):
        import zipfile, io
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(dest)
    else:
        (dest / filename).write_bytes(content)


# ── agent runner (runs in background thread) ──────────────────────────────────

def _agent_thread(
    run_id: str,
    repo_root: str,
    task_description: str,
    llm_provider: Optional[str],
    llm_base_url: Optional[str],
    llm_model: Optional[str],
    tmp_dir: str,
    session_id: Optional[str] = None,
) -> None:
    q = _runs[run_id]["queue"]

    def emit(event_type: str, data: dict) -> None:
        q.put_nowait({"type": event_type, "data": data})

    try:
        from main import run_agent
        from llm import create_local_llm_client

        client = None
        if llm_provider:
            client = create_local_llm_client(
                provider=llm_provider,
                base_url=llm_base_url or None,
                model=llm_model or "gpt-4o-mini",
            )

        trace_path = os.path.join(tmp_dir, "trace.jsonl")

        state = run_agent(
            task_description=task_description,
            repo_root=repo_root,
            client=client,
            trace_path=trace_path,
            run_id=run_id,
            step_callback=emit,
        )

        _runs[run_id]["result"] = {
            "validation": state.validation_status.value,
            "run_status": state.run_status.value,
            "stop_reason": state.stop_reason,
            "replan_count": state.replan_count,
            "steps_completed": sum(1 for s in state.plan if s.status.value == "completed"),
            "steps_total": len(state.plan),
            "files_modified": state.files_modified,
            "files_read": state.files_read,
            "final_answer": state.final_answer or "",
            "elapsed_s": round((state.finished_at or time.time()) - state.started_at, 2)
            if state.started_at else 0,
            "plan": [
                {"step_id": s.step_id, "task": s.task, "status": s.status.value}
                for s in state.plan
            ],
        }

    except Exception as exc:
        emit("error", {"message": str(exc)})
        _runs[run_id]["result"] = {"error": str(exc)}
    finally:
        _runs[run_id]["done"] = True
        q.put_nowait(None)  # sentinel

        # Add agent summary to session history
        if session_id and session_id in _sessions:
            result = _runs[run_id].get("result") or {}
            answer = result.get("final_answer", "")
            if answer:
                _sessions[session_id]["history"].append({"role": "agent", "content": answer})

        # Only clean up workspace for one-shot runs (sessions manage their own lifecycle)
        if not session_id:
            def _cleanup():
                time.sleep(300)
                shutil.rmtree(tmp_dir, ignore_errors=True)
            threading.Thread(target=_cleanup, daemon=True).start()


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/run")
async def start_run(
    repo_url: Optional[str] = Form(None),
    task: Optional[str] = Form(None),
    llm_provider: Optional[str] = Form(None),
    llm_base_url: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if not repo_url and not file:
        raise HTTPException(400, "Provide either repo_url or a file upload")

    run_id = str(uuid.uuid4())[:8]
    tmp_dir = tempfile.mkdtemp(prefix=f"agent_{run_id}_")
    repo_root = os.path.join(tmp_dir, "repo")

    # Set up workspace
    try:
        if repo_url and repo_url.strip():
            _clone_repo(repo_url.strip(), Path(repo_root))
        elif file:
            content = await file.read()
            _write_uploaded_file(content, file.filename or "upload.py", Path(repo_root))
        else:
            raise HTTPException(400, "No input provided")
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, str(exc))

    task_description = (task or "").strip() or (
        "Analyze this repository for bugs. Fix any bugs you find and run the tests to verify. "
        "If no bugs are found, summarize what the code does and suggest improvements."
    )

    _runs[run_id] = {
        "queue": queue.Queue(),
        "done": False,
        "result": None,
        "task": task_description,
        "repo_url": repo_url,
    }

    thread = threading.Thread(
        target=_agent_thread,
        args=(run_id, repo_root, task_description, llm_provider, llm_base_url, llm_model, tmp_dir),
        daemon=True,
    )
    thread.start()

    return {"run_id": run_id}


@app.get("/api/run/{run_id}/stream")
async def stream_run(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")

    async def event_generator():
        q = _runs[run_id]["queue"]
        # Yield a comment to open the connection immediately
        yield ": connected\n\n"

        while True:
            # Poll the thread-safe queue from async context
            try:
                item = q.get_nowait()
            except queue.Empty:
                if _runs[run_id]["done"] and q.empty():
                    break
                await asyncio.sleep(0.15)
                continue

            if item is None:  # sentinel
                break

            yield f"data: {json.dumps(item)}\n\n"

        yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/run/{run_id}")
async def get_run(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    return JSONResponse({
        "run_id": run_id,
        "done": run["done"],
        "result": run["result"],
    })


# ── Session endpoints (multi-turn chat) ──────────────────────────────────────

@app.post("/api/session")
async def create_session(
    repo_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """Clone the repo once; returns session_id for follow-up messages."""
    if not repo_url and not file:
        raise HTTPException(400, "Provide either repo_url or a file upload")

    session_id = str(uuid.uuid4())[:8]
    tmp_dir = tempfile.mkdtemp(prefix=f"sess_{session_id}_")
    repo_root = os.path.join(tmp_dir, "repo")
    repo_label = ""

    try:
        if repo_url and repo_url.strip():
            repo_label = repo_url.strip()
            _clone_repo(repo_label, Path(repo_root))
        elif file:
            content = await file.read()
            repo_label = file.filename or "upload"
            _write_uploaded_file(content, repo_label, Path(repo_root))
        else:
            raise HTTPException(400, "No input provided")
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, str(exc))

    _sessions[session_id] = {
        "tmp_dir": tmp_dir,
        "repo_root": repo_root,
        "repo_label": repo_label,
        "history": [],
        "created_at": time.time(),
        "last_active": time.time(),
    }

    def _session_watchdog():
        while True:
            time.sleep(300)
            s = _sessions.get(session_id)
            if not s:
                break
            if time.time() - s["last_active"] > 1800:  # 30 min idle → clean up
                shutil.rmtree(s["tmp_dir"], ignore_errors=True)
                _sessions.pop(session_id, None)
                break

    threading.Thread(target=_session_watchdog, daemon=True).start()
    return {"session_id": session_id, "repo_label": repo_label}


@app.post("/api/session/{session_id}/message")
async def session_message(
    session_id: str,
    task: str = Form(...),
    llm_provider: Optional[str] = Form(None),
    llm_base_url: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
):
    """Send a follow-up message in an existing session. Returns run_id for SSE streaming."""
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found or expired (idle >30 min)")

    session = _sessions[session_id]
    session["last_active"] = time.time()

    task_with_context = _build_contextual_task(task, session["history"])
    session["history"].append({"role": "user", "content": task})

    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = {
        "queue": queue.Queue(),
        "done": False,
        "result": None,
        "task": task,
        "session_id": session_id,
    }

    thread = threading.Thread(
        target=_agent_thread,
        args=(run_id, session["repo_root"], task_with_context,
              llm_provider, llm_base_url, llm_model, session["tmp_dir"], session_id),
        daemon=True,
    )
    thread.start()
    return {"run_id": run_id}


@app.delete("/api/session/{session_id}")
async def close_session(session_id: str):
    """Explicitly end a session and clean up its workspace."""
    s = _sessions.pop(session_id, None)
    if s:
        shutil.rmtree(s["tmp_dir"], ignore_errors=True)
    return {"closed": session_id}


# ── [DB PLACEHOLDER] history endpoints ────────────────────────────────────────
# Wire these up once _db_pool is initialised above.

@app.get("/api/history")
async def list_history(limit: int = 20, offset: int = 0):
    """
    [DB PLACEHOLDER] Return paginated run history from PostgreSQL.

    When DB is connected:
        rows = await _db_pool.fetch(
            "SELECT run_id, task, status, created_at, finished_at "
            "FROM runs ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit, offset,
        )
        return [dict(r) for r in rows]
    """
    raise HTTPException(501, "History endpoint requires PostgreSQL. See DB PLACEHOLDER in api/server.py.")


@app.delete("/api/run/{run_id}")
async def delete_run(run_id: str):
    """
    [DB PLACEHOLDER] Delete a run record from PostgreSQL.

    When DB is connected:
        await _db_pool.execute("DELETE FROM runs WHERE run_id = $1", run_id)
        return {"deleted": run_id}
    """
    _runs.pop(run_id, None)
    raise HTTPException(501, "Persistent delete requires PostgreSQL. In-memory entry removed.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8080, reload=False)
