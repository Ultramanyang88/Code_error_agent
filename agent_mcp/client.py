from __future__ import annotations

import asyncio
import subprocess
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolClient:
    """Wraps an MCP server process and exposes its tools as callables."""

    def __init__(self, command: str, args: List[str], namespace: str):
        self.command = command
        self.args = args
        self.namespace = namespace  # e.g. "mcp_fs" → tool names become "mcp_fs__read_file"
        self._tools: Dict[str, Any] = {}

    async def _fetch_tools(self) -> Dict[str, Any]:
        params = StdioServerParameters(command=self.command, args=self.args)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                result = await session.list_tools()
                return {t.name: t for t in result.tools}

    async def _call_tool(self, tool_name: str, arguments: dict) -> str:
        params = StdioServerParameters(command=self.command, args=self.args)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    def get_tool_map(self) -> Dict[str, Any]:
        """Return sync callables keyed by namespaced tool name, for use in get_tool_map()."""
        tools_meta = asyncio.run(self._fetch_tools())
        tool_map = {}
        for name in tools_meta:
            namespaced = f"{self.namespace}__{name}"
            def make_fn(n):
                def fn(state, **kwargs):
                    from core.state import ToolResult
                    try:
                        output = asyncio.run(self._call_tool(n, kwargs))
                        return ToolResult(tool_name=namespaced, success=True, output=output)
                    except Exception as e:
                        return ToolResult(tool_name=namespaced, success=False, output="", error=str(e))
                fn.__doc__ = tools_meta[n].description
                return fn
            tool_map[namespaced] = make_fn(name)
        return tool_map