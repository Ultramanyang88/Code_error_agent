from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio
import sys

sys.path.insert(0, ".")

from core.state import AgentState
from tools.tools import get_tool_map
from tools.specs import TOOL_SPECS

app = Server("coding-agent-tools")
_state: AgentState | None = None

def get_state() -> AgentState:
    global _state
    if _state is None:
        _state = AgentState(input_query="mcp", repo_root=".")
    return _state

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    tools = []
    for name, spec in TOOL_SPECS.items():
        params = spec.get("parameters", {})
        properties = {
            k: {"type": v.get("type", "string"), "description": v.get("description", "")}
            for k, v in params.items()
        }
        required = [k for k, v in params.items() if v.get("required")]
        tools.append(types.Tool(
            name=name,
            description=spec.get("description", ""),
            inputSchema={"type": "object", "properties": properties, "required": required},
        ))
    return tools

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    tool_map = get_tool_map()
    if name not in tool_map:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    result = tool_map[name](state=get_state(), **arguments)
    return [types.TextContent(type="text", text=result.to_text())]

async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
