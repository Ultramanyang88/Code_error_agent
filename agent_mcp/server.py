from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp import types
from pydantic import AnyUrl
import asyncio
import sys

sys.path.insert(0, ".")

from core.state import AgentState
from tools.tools import get_tool_map
from tools.specs import TOOL_SPECS, TOOL_CATEGORIES, category_summary_text
from skills.registry import SkillRegistry

app = Server("coding-agent-tools")
_state: AgentState | None = None
_skills: SkillRegistry | None = None

def get_state() -> AgentState:
    global _state
    if _state is None:
        _state = AgentState(input_query="mcp", repo_root=".")
    return _state

def get_skills() -> SkillRegistry:
    global _skills
    if _skills is None:
        _skills = SkillRegistry(skills_dir="skills")
    return _skills

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


# ── resources ────────────────────────────────────────────────────────────────
#
# Two browsable/addressable collections: one resource per tool category
# (tool-category://<name>) and one per skill playbook (skill://<name>).
# These are read-only, static-ish reference material -- the kind of thing a
# resource is for, as opposed to retrieve_context's arbitrary-query search,
# which stays a *tool* since its output depends on caller-supplied input.

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    resources = []

    for category, purpose in TOOL_CATEGORIES.items():
        resources.append(types.Resource(
            uri=AnyUrl(f"tool-category://{category}"),
            name=f"Tool category: {category}",
            description=purpose,
            mimeType="text/plain",
        ))

    for skill in get_skills().skills:
        resources.append(types.Resource(
            uri=AnyUrl(f"skill://{skill.name}"),
            name=f"Skill: {skill.name}",
            description=skill.summary,
            mimeType="text/markdown",
        ))

    return resources

@app.read_resource()
async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
    scheme = uri.scheme
    identifier = str(uri).split("://", 1)[-1]

    if scheme == "tool-category":
        if identifier not in TOOL_CATEGORIES:
            raise ValueError(f"Unknown tool category: {identifier}")
        return [ReadResourceContents(content=category_summary_text(identifier), mime_type="text/plain")]

    if scheme == "skill":
        skill = next((s for s in get_skills().skills if s.name == identifier), None)
        if skill is None:
            raise ValueError(f"Unknown skill: {identifier}")
        content = f"Skill: {skill.name}\nSummary: {skill.summary}\n\n{skill.full_text}"
        return [ReadResourceContents(content=content, mime_type="text/markdown")]

    raise ValueError(f"Unsupported resource URI scheme: {scheme}")


# ── prompts ──────────────────────────────────────────────────────────────────
#
# Every skill playbook (skills/*.md), exposed as a standard MCP prompt so an
# external host can pull in "how do I fix an import error" etc. as a reusable
# prompt template instead of having to know this repo's skill-file format.

@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name=skill.name,
            description=skill.summary,
            arguments=[
                types.PromptArgument(
                    name="task",
                    description="The current task/step description to apply this playbook to.",
                    required=False,
                )
            ],
        )
        for skill in get_skills().skills
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
    skill = next((s for s in get_skills().skills if s.name == name), None)
    if skill is None:
        raise ValueError(f"Unknown prompt/skill: {name}")

    task = (arguments or {}).get("task")
    header = f"Task: {task}\n\n" if task else ""
    text = f"{header}Skill: {skill.name}\nSummary: {skill.summary}\n\nProcedure:\n{skill.full_text}"

    return types.GetPromptResult(
        description=skill.summary,
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=text),
            )
        ],
    )


async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
