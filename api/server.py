from __future__ import annotations

import asyncio
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    # Explicit path (not just load_dotenv()'s default upward search from
    # CWD): this also needs to work when an `rq worker` process imports this
    # module to resolve a job function, and that process's CWD isn't
    # guaranteed to be the repo root the way `python api/server.py` usually is.
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from db import store as db_store  # noqa: E402 - needs PROJECT_ROOT on sys.path first
from db import cache as db_cache  # noqa: E402
from db.queue import is_queue_enabled  # noqa: E402
from sandbox.docker_sandbox import is_sandbox_enabled, is_docker_available  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional: only activates if $DATABASE_URL/$REDIS_URL are set and
    # reachable. The app's in-memory _sessions/_runs dicts remain the source
    # of truth for the live request/SSE path either way -- see db/store.py's
    # module docstring for why Postgres is write-through, not a replacement,
    # and db/cache.py's for how Redis fits in (session TTL + SSE relay for
    # out-of-process RQ jobs).
    await asyncio.to_thread(db_store.init_pool)
    await asyncio.to_thread(db_cache.init_redis)
    print("[*] Postgres persistence: " + ("enabled" if db_store.available() else "disabled (in-memory only)"))
    print("[*] Redis: " + ("enabled" if db_cache.available() else "disabled (thread-based fallbacks)"))
    print("[*] Task queue (RQ): " + (
        "enabled" if is_queue_enabled() and db_cache.available() else "disabled (in-process threads)"
    ))
    yield
    await asyncio.to_thread(db_store.close_pool)
    await asyncio.to_thread(db_cache.close_redis)


app = FastAPI(title="Code Error Agent", lifespan=lifespan)

# Serve static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory run registry: run_id → {queue, done, result}
_runs: Dict[str, Dict[str, Any]] = {}

# Session registry: session_id → {tmp_dir, repo_root, repo_label, history, last_active}
_sessions: Dict[str, Dict[str, Any]] = {}

_AGENT_TASK_KEYWORDS = {
    "fix", "implement", "modify", "change", "update", "patch",
    "debug", "solve", "add code", "complete code", "run the test", "run tests",
}

def _needs_full_agent_run(message: str) -> bool:
    lower = message.lower()
    return any(kw in lower for kw in _AGENT_TASK_KEYWORDS)

@app.post("/api/session/{session_id}/chat")
async def session_chat_stream(
    session_id: str,
    message: str = Form(...),
    llm_provider: Optional[str] = Form(None),
    llm_base_url: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found or expired")

    session = _sessions[session_id]
    session["last_active"] = time.time()
    session["history"].append({"role": "user", "content": message})
    await asyncio.to_thread(db_store.touch_session, session_id)
    await asyncio.to_thread(db_store.add_message, session_id, "user", message)
    db_cache.touch_session_ttl(session_id)

    q: "queue.Queue" = queue.Queue()

    # 轻量聊天只给只读工具：允许模型自己读仓库找答案，但不能悄悄改代码/跑命令。
    _LIGHT_CHAT_TOOLS = {"list_files", "read_file", "search_code", "retrieve_context", "git_diff"}
    _MAX_TOOL_ROUNDS = 4

    def _worker():
        from llm import create_local_llm_client
        from main import _bound_task_description
        from tools.tools import get_tool_map
        from tools.specs import to_openai_tools
        from core.state import AgentState, ToolResult

        client = create_local_llm_client(
            provider=llm_provider or "openai_compatible",
            base_url=llm_base_url or None,
            model=llm_model or "gpt-4o-mini",
        )

        all_tools = get_tool_map()
        tools = {name: fn for name, fn in all_tools.items() if name in _LIGHT_CHAT_TOOLS}
        tool_schema = to_openai_tools(list(tools.keys()))
        fake_state = AgentState(input_query=message, repo_root=session["repo_root"])

        bounded_message = _bound_task_description(message)
        history_text = "\n".join(
            f"{'User' if m['role']=='user' else 'Agent'}: {_bound_task_description(m['content'])}"
            for m in session["history"][-8:]
        )
        messages: list[Dict[str, Any]] = [
            {"role": "system", "content": (
                "You are a grounded coding assistant answering questions about the repository "
                "checked out at the current working directory. Use the available tools "
                "(list_files, read_file, search_code, retrieve_context, git_diff) to inspect the "
                "real code before answering — never guess. Once you have enough evidence, answer "
                "directly in plain text without calling more tools."
            )},
            {"role": "user", "content": f"{history_text}\n\nUser: {bounded_message}"},
        ]

        full_reply = []
        try:
            for _ in range(_MAX_TOOL_ROUNDS):
                response = client.chat(messages, tools=tool_schema)
                raw_calls = response.get("tool_calls") if isinstance(response, dict) else None

                if not raw_calls:
                    for delta in client.chat_stream(messages):
                        full_reply.append(delta)
                        q.put_nowait({"delta": delta})
                    break

                call = raw_calls[0]
                fn = call.get("function", {})
                tool_name = fn.get("name")
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                tool_fn = tools.get(tool_name)
                if tool_fn:
                    result = tool_fn(state=fake_state, **arguments)
                    result_text = result.to_text(max_chars=2000) if isinstance(result, ToolResult) else str(result)
                else:
                    result_text = f"Tool not available: {tool_name}"

                messages.append({"role": "assistant", "content": response.get("content"), "tool_calls": [call]})
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result_text})
            else:
                # 工具轮数用完还没给出最终答案：强制来一轮不带 tools 的流式收尾。
                for delta in client.chat_stream(messages):
                    full_reply.append(delta)
                    q.put_nowait({"delta": delta})
        except Exception as exc:
            q.put_nowait({"error": f"LLM unreachable ({type(exc).__name__}): {exc}"})
        finally:
            if full_reply:
                reply_text = "".join(full_reply)
                session["history"].append({"role": "agent", "content": reply_text})
                db_store.add_message(session_id, "agent", reply_text)
            q.put_nowait(None)

    threading.Thread(target=_worker, daemon=True).start()

    async def event_generator():
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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

    # Sandboxed clone: an ephemeral, network-enabled container (git clone is
    # the one operation that legitimately needs network) -- distinct from
    # the persistent, network-*disabled* sandbox this same repo_root gets
    # later for run_command/apply_patch (see sandbox/docker_sandbox.py's
    # module docstring for why those are two different containers).
    if is_sandbox_enabled() and is_docker_available():
        try:
            from sandbox.docker_sandbox import run_ephemeral
            exit_code, _, stderr = run_ephemeral(
                cwd=str(dest),
                command=f"git clone --depth 1 {shlex.quote(url)} .",
                network="bridge",
                timeout=120,
            )
            if exit_code != 0:
                raise RuntimeError(f"git clone failed:\n{stderr[:800]}")
            return
        except Exception as e:
            print(f"[!] Sandboxed clone failed, falling back to host clone: {e}")

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


def _close_mcp_for_repo(repo_root: str) -> None:
    """
    Close the shared MCP filesystem client (if any) that was opened for this
    repo_root, so its `npx` subprocess doesn't stay alive pointed at a
    workspace directory we're about to delete. Safe to call even if no MCP
    client was ever opened for this repo (e.g. `npx` unavailable) — best-effort.
    """
    try:
        from agent_mcp.client import close_shared_mcp_client
        close_shared_mcp_client(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "."],
            namespace="mcp_fs",
            cwd=str(Path(repo_root).resolve()),
        )
    except Exception:
        pass


def _close_sandbox_for_repo(repo_root: str) -> None:
    """
    Stop and remove this repo's persistent sandbox container (if any) before
    its workspace directory is deleted. Safe to call even if sandboxing was
    never enabled/no container was ever started for this repo -- best-effort.
    """
    try:
        from sandbox.docker_sandbox import close_shared_sandbox
        close_shared_sandbox(repo_root)
    except Exception:
        pass


# ── agent runner ────────────────────────────────────────────────────────────
#
# Split in two on purpose:
#   _execute_agent_run   -- process-agnostic core. Touches only Redis/Postgres
#                            (safe from any process) and its own return value;
#                            never the API process's in-memory _runs/_sessions.
#                            This is what runs as a plain thread (default) OR
#                            as an RQ job in a separate `rq worker` process
#                            (opt-in, see db/queue.py) -- same function either
#                            way, since it doesn't know or care which.
#   _dispatch_agent_run   -- entry point the routes call. Owns the bookkeeping
#                            that HAS to live in this process (the local SSE
#                            queue, session history, workspace cleanup
#                            scheduling) and picks how the core function runs.

def _execute_agent_run(
    run_id: str,
    repo_root: str,
    task_description: str,
    llm_provider: Optional[str],
    llm_base_url: Optional[str],
    llm_model: Optional[str],
    tmp_dir: str,
    session_id: Optional[str] = None,
    repo_url: Optional[str] = None,
    extra_emit=None,
) -> dict:
    # When this runs as an RQ job, it's executing in a separate `rq worker`
    # OS process that never went through api/server.py's FastAPI `lifespan`
    # startup -- db_cache/db_store would still be un-initialized there
    # (their module-level connection is per-process). init_redis()/
    # init_pool() are both idempotent (return immediately if already
    # connected), so this is a cheap no-op on the in-process/thread path and
    # the actual connection step the first time this runs inside a worker.
    # The worker process still needs $REDIS_URL/$DATABASE_URL set in its own
    # environment for this to have anything to connect to -- see README.
    db_cache.init_redis()
    db_store.init_pool()

    def emit(event_type: str, data: dict) -> None:
        event = {"type": event_type, "data": data}
        db_cache.publish_event(run_id, event)  # no-op if Redis isn't configured
        if extra_emit:
            extra_emit(event)

    db_store.create_run(run_id, session_id, task_description, repo_url)

    result: dict = {}
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
            session_id=session_id,
        )

        result = {
            "validation": state.validation_status.value,
            "run_status": state.run_status.value,
            "stop_reason": state.stop_reason,
            "replan_count": state.replan_count,
            "tool_call_count": len(state.tool_history),
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
        result = {"error": str(exc), "run_status": "failed"}

    finally:
        db_store.finish_run(run_id, result)
        # Published directly (bypassing emit()/extra_emit): this is an
        # internal signal for _dispatch_agent_run's relay/bookkeeping, not a
        # frontend-facing event. The in-process path doesn't need it (it
        # already gets `result` as this function's return value); the RQ
        # relay path uses it as the terminal marker -- see db/cache.py's
        # subscribe_events().
        db_cache.publish_event(run_id, {"type": "run_result", "data": result})

    return result


def _dispatch_agent_run(
    run_id: str,
    repo_root: str,
    task_description: str,
    llm_provider: Optional[str],
    llm_base_url: Optional[str],
    llm_model: Optional[str],
    tmp_dir: str,
    session_id: Optional[str] = None,
) -> None:
    repo_url = _runs[run_id].get("repo_url") or (
        _sessions.get(session_id, {}).get("repo_label") if session_id else None
    )

    def _finalize(result: dict) -> None:
        run = _runs.get(run_id)
        if run is None:
            return
        run["result"] = result
        run["done"] = True
        run["queue"].put_nowait(None)  # sentinel for the SSE poller

        if session_id and session_id in _sessions:
            answer = result.get("final_answer", "")
            if answer:
                _sessions[session_id]["history"].append({"role": "agent", "content": answer})
                _sessions[session_id]["last_retrieved_context"] = answer
                db_store.add_message(session_id, "agent", answer, run_id=run_id)

        # Only clean up workspace for one-shot runs (sessions manage their own lifecycle)
        if not session_id:
            def _cleanup():
                time.sleep(300)
                _close_mcp_for_repo(repo_root)
                _close_sandbox_for_repo(repo_root)
                shutil.rmtree(tmp_dir, ignore_errors=True)
            threading.Thread(target=_cleanup, daemon=True).start()

    if is_queue_enabled() and db_cache.available():
        def _run_via_queue():
            try:
                from db.queue import enqueue_run
                enqueue_run(
                    _execute_agent_run, run_id, repo_root, task_description,
                    llm_provider, llm_base_url, llm_model, tmp_dir, session_id, repo_url,
                    job_id=run_id,
                )
            except Exception as e:
                _runs[run_id]["queue"].put_nowait({"type": "error", "data": {"message": f"Failed to enqueue run: {e}"}})
                _finalize({"error": str(e), "run_status": "failed"})
                return

            result: dict = {}
            for event in db_cache.subscribe_events(run_id):
                if event.get("type") == "run_result":
                    result = event.get("data", {})
                    continue  # internal marker, not a frontend-facing SSE event
                _runs[run_id]["queue"].put_nowait(event)
            _finalize(result)

        threading.Thread(target=_run_via_queue, daemon=True).start()
    else:
        def _run_in_thread():
            result = _execute_agent_run(
                run_id, repo_root, task_description,
                llm_provider, llm_base_url, llm_model, tmp_dir, session_id, repo_url,
                extra_emit=lambda event: _runs[run_id]["queue"].put_nowait(event),
            )
            _finalize(result)

        threading.Thread(target=_run_in_thread, daemon=True).start()


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

    _dispatch_agent_run(run_id, repo_root, task_description, llm_provider, llm_base_url, llm_model, tmp_dir)

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
    await asyncio.to_thread(db_store.create_session, session_id, repo_label, repo_root)
    db_cache.touch_session_ttl(session_id)

    def _session_watchdog():
        while True:
            time.sleep(300)
            s = _sessions.get(session_id)
            if not s:
                break
            # Redis TTL when available (an EXPIRE check, not a timestamp
            # comparison against something only this process updates -- the
            # same signal would work if session activity were ever tracked
            # from more than one process). Falls back to the original
            # last_active math when Redis isn't configured.
            redis_active = db_cache.is_session_active(session_id)
            expired = (not redis_active) if redis_active is not None else (time.time() - s["last_active"] > 1800)
            if expired:
                db_store.close_session(session_id)
                db_cache.clear_session_ttl(session_id)
                _close_mcp_for_repo(s["repo_root"])
                _close_sandbox_for_repo(s["repo_root"])
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
    db_cache.touch_session_ttl(session_id)

    task_with_context = _build_contextual_task(task, session["history"])
    session["history"].append({"role": "user", "content": task})
    await asyncio.to_thread(db_store.touch_session, session_id)
    await asyncio.to_thread(db_store.add_message, session_id, "user", task)

    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = {
        "queue": queue.Queue(),
        "done": False,
        "result": None,
        "task": task,
        "session_id": session_id,
    }

    _dispatch_agent_run(
        run_id, session["repo_root"], task_with_context,
        llm_provider, llm_base_url, llm_model, session["tmp_dir"], session_id,
    )
    return {"run_id": run_id}


@app.delete("/api/session/{session_id}")
async def close_session(session_id: str):
    """Explicitly end a session and clean up its workspace."""
    s = _sessions.pop(session_id, None)
    if s:
        await asyncio.to_thread(db_store.close_session, session_id)
        db_cache.clear_session_ttl(session_id)
        _close_mcp_for_repo(s["repo_root"])
        _close_sandbox_for_repo(s["repo_root"])
        shutil.rmtree(s["tmp_dir"], ignore_errors=True)
    return {"closed": session_id}


# ── history / persistence-backed endpoints ────────────────────────────────────
# All of these require $DATABASE_URL to be set and reachable (see db/store.py,
# db/schema.sql). Without it they 501 -- the in-memory-only dicts above have
# no durable history to serve from a fresh process.

@app.get("/api/history")
async def list_history(limit: int = 20, offset: int = 0):
    """Paginated run history from Postgres."""
    if not db_store.available():
        raise HTTPException(501, "History requires Postgres. Set $DATABASE_URL (see docker-compose.yml).")
    return await asyncio.to_thread(db_store.list_runs, limit, offset)


@app.get("/api/sessions")
async def list_sessions(limit: int = 20, offset: int = 0):
    """Paginated session history from Postgres (for a session sidebar)."""
    if not db_store.available():
        raise HTTPException(501, "Session history requires Postgres. Set $DATABASE_URL (see docker-compose.yml).")
    return await asyncio.to_thread(db_store.list_sessions, limit, offset)


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 200):
    """
    Full message history for a session from Postgres -- lets the frontend
    reload a conversation after a page refresh, not just while _sessions
    still holds it in memory (which is lost on process restart).
    """
    if not db_store.available():
        raise HTTPException(501, "Session history requires Postgres. Set $DATABASE_URL (see docker-compose.yml).")
    return await asyncio.to_thread(db_store.get_messages, session_id, limit)


@app.delete("/api/run/{run_id}")
async def delete_run(run_id: str):
    """Delete a run record. Removes the in-memory entry regardless; the persisted row only if Postgres is configured."""
    _runs.pop(run_id, None)
    if not db_store.available():
        raise HTTPException(501, "Persistent delete requires Postgres. In-memory entry removed.")
    deleted = await asyncio.to_thread(db_store.delete_run, run_id)
    return {"deleted": run_id, "persisted_row_removed": deleted}


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8080, reload=False)
