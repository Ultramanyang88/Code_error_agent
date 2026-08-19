from __future__ import annotations

import asyncio
import atexit
import threading
from typing import Any, Coroutine, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolClient:
    """
    Wraps an MCP server process and exposes its tools as sync callables.

    One persistent connection (one subprocess, one handshake) is kept alive in
    a dedicated background thread for the lifetime of this client, instead of
    spawning a fresh subprocess + re-running the MCP handshake on every single
    tool call (or every time a caller wants a fresh tool map). Call connect()
    once — or just call get_tool_map(), which connects lazily — and close()
    when you're done with it. Prefer get_shared_mcp_client() below over
    constructing this directly, so repeated callers reuse one connection.
    """

    def __init__(
        self,
        command: str,
        args: List[str],
        namespace: str,
        cwd: Optional[str] = None,
        connect_timeout: float = 30.0,
    ):
        self.command = command
        self.args = args
        self.namespace = namespace  # e.g. "mcp_fs" → tool names become "mcp_fs__read_file"
        self.cwd = cwd  # working directory the subprocess is spawned in (e.g. so "." in args resolves correctly)
        self.connect_timeout = connect_timeout

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._stop_future: Optional["asyncio.Future[None]"] = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._connect_error: Optional[BaseException] = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Start the background loop/thread and block until the MCP handshake completes."""
        if self._thread is not None:
            return  # already connected (or a previous connect() already ran)

        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name=f"mcp-{self.namespace}"
        )
        self._thread.start()

        if not self._ready.wait(timeout=self.connect_timeout):
            raise TimeoutError(
                f"MCP server '{self.namespace}' ({self.command}) did not become "
                f"ready within {self.connect_timeout}s"
            )

        if self._connect_error is not None:
            raise self._connect_error

    def close(self) -> None:
        """Signal the background session to shut down and wait for it to exit."""
        if self._thread is None or self._loop is None or self._closed.is_set():
            return
        self._closed.set()
        loop = self._loop

        def _signal_stop() -> None:
            if self._stop_future is not None and not self._stop_future.done():
                self._stop_future.set_result(None)

        loop.call_soon_threadsafe(_signal_stop)
        self._thread.join(timeout=10)
        self._thread = None

    def __enter__(self) -> "MCPToolClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        return self._session is not None and self._loop is not None

    # ── background event loop ────────────────────────────────────────────

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._session_lifecycle())
        except BaseException as exc:  # noqa: BLE001 - surfaced to connect() below
            if not self._ready.is_set():
                self._connect_error = exc
                self._ready.set()
        finally:
            self._session = None
            loop.close()

    async def _session_lifecycle(self) -> None:
        """Open the subprocess + MCP session once, then idle until close() fires."""
        params = StdioServerParameters(command=self.command, args=self.args, cwd=self.cwd)
        self._stop_future = asyncio.get_running_loop().create_future()

        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._stop_future

    # ── tool calls (run on the background loop, awaited synchronously) ─────

    async def _fetch_tools_async(self) -> Dict[str, Any]:
        if self._session is None:
            raise RuntimeError(f"MCP client '{self.namespace}' session is not ready.")
        result = await self._session.list_tools()
        return {t.name: t for t in result.tools}

    async def _call_tool_async(self, tool_name: str, arguments: dict) -> str:
        if self._session is None:
            raise RuntimeError(f"MCP client '{self.namespace}' session is not ready.")
        result = await self._session.call_tool(tool_name, arguments)
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    def _run_coro_sync(self, coro: Coroutine[Any, Any, Any]) -> Any:
        if self._loop is None:
            raise RuntimeError(f"MCP client '{self.namespace}' is not connected — call connect() first.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.connect_timeout)

    def _make_tool_fn(self, tool_name: str, namespaced_name: str, description: str):
        def fn(state, **kwargs):
            from core.state import ToolResult
            try:
                output = self._run_coro_sync(self._call_tool_async(tool_name, kwargs))
                return ToolResult(tool_name=namespaced_name, success=True, output=output)
            except Exception as e:
                return ToolResult(tool_name=namespaced_name, success=False, output="", error=str(e))

        fn.__doc__ = description
        return fn

    def get_tool_map(self) -> Dict[str, Any]:
        """Return sync callables keyed by namespaced tool name, for use in get_tool_map()."""
        if not self.is_connected:
            self.connect()

        tools_meta = self._run_coro_sync(self._fetch_tools_async())
        tool_map: Dict[str, Any] = {}
        for name, meta in tools_meta.items():
            namespaced = f"{self.namespace}__{name}"
            tool_map[namespaced] = self._make_tool_fn(name, namespaced, getattr(meta, "description", "") or "")
        return tool_map


# ── process-wide connection cache ────────────────────────────────────────────
#
# Without this, every caller that wants an MCP tool map (every run_agent()
# call — i.e. every chat turn) would build its own MCPToolClient and pay a
# fresh subprocess spawn + MCP handshake. get_shared_mcp_client() hands back
# one persistent, already-connected client per (command, args, namespace, cwd),
# shared across the whole process.
#
# cwd is part of the cache key (not just an arg) on purpose: a filesystem-style
# MCP server resolves its "." argument against the subprocess's own working
# directory, and each session/repo needs its own subprocess pointed at its own
# repo_root — sharing one cached client across two different repos would let
# one session's mcp_fs__* tools silently read/write the wrong tree.

_shared_clients: Dict[Tuple[str, Tuple[str, ...], str, Optional[str]], MCPToolClient] = {}
_shared_clients_lock = threading.Lock()


def get_shared_mcp_client(
    command: str,
    args: List[str],
    namespace: str,
    cwd: Optional[str] = None,
    connect_timeout: float = 30.0,
) -> MCPToolClient:
    """
    Return a process-wide MCPToolClient for (command, args, namespace, cwd),
    connecting it once and reusing the same persistent subprocess/session on
    every subsequent call instead of reconnecting each time.

    Raises whatever connect() raises on the first call for a given key; a
    failed connection is never cached, so the next call retries from scratch.
    """
    key = (command, tuple(args), namespace, cwd)

    with _shared_clients_lock:
        existing = _shared_clients.get(key)
        if existing is not None:
            return existing

    candidate = MCPToolClient(
        command=command, args=args, namespace=namespace, cwd=cwd, connect_timeout=connect_timeout
    )
    candidate.connect()  # may raise — caller decides how to handle a down/missing MCP server

    with _shared_clients_lock:
        # Another thread may have connected one for the same key concurrently;
        # keep whichever landed first and close the redundant one.
        winner = _shared_clients.setdefault(key, candidate)
        if winner is not candidate:
            candidate.close()
        return winner


def close_shared_mcp_client(
    command: str,
    args: List[str],
    namespace: str,
    cwd: Optional[str] = None,
) -> None:
    """
    Evict and close the cached client for one (command, args, namespace, cwd) key.

    Callers that spin up a per-session/per-repo MCP client via
    get_shared_mcp_client(cwd=...) should call this when that session/workspace
    is torn down — otherwise the subprocess and its background thread stay
    alive for the lifetime of the process, pointed at a directory that may no
    longer exist.
    """
    key = (command, tuple(args), namespace, cwd)
    with _shared_clients_lock:
        client = _shared_clients.pop(key, None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def close_all_shared_mcp_clients() -> None:
    """Close every cached client. Safe to call multiple times."""
    with _shared_clients_lock:
        clients = list(_shared_clients.values())
        _shared_clients.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            pass


atexit.register(close_all_shared_mcp_clients)
