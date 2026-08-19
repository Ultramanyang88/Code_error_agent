"""
Unified view of "what tools exist and what category is each in", merging
the static built-in taxonomy (specs.py's TOOL_SPECS/TOOL_CATEGORIES) with
categories declared for dynamically-discovered MCP tools
(agent_mcp/config.py's MCPServerConfig.category/tool_categories) -- without
this, an MCP tool like "mcp_fs__read_file" is invisible to
expand_by_category() and the planner has no compact way to reason about it.
Also gives the planner a category-level summary instead of one line per
tool, keeping prompt size roughly constant regardless of how many MCP
servers/tools get registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .specs import TOOL_CATEGORIES, category_of as _static_category_of


@dataclass
class ToolRegistry:
    tools: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    # tool name -> category, for tools tools/specs.py doesn't know about (MCP)
    extra_categories: Dict[str, str] = field(default_factory=dict)
    # category -> one-line purpose, for categories MCP servers introduce that
    # aren't already in TOOL_CATEGORIES
    extra_category_purpose: Dict[str, str] = field(default_factory=dict)

    def category_of(self, tool_name: str) -> Optional[str]:
        return _static_category_of(tool_name) or self.extra_categories.get(tool_name)

    def all_category_purpose(self) -> Dict[str, str]:
        merged = dict(TOOL_CATEGORIES)
        merged.update(self.extra_category_purpose)
        return merged

    def tools_in_categories(self, categories: List[str]) -> List[str]:
        return [name for name in self.tools if self.category_of(name) in categories]

    def expand_by_category(self, seed: List[str]) -> List[str]:
        """
        Expand a plan step's suggested_tools seed into concrete tool names,
        pulling in every tool that shares a category with the seed. The seed
        may mix concrete names ("write_file") and bare category names
        ("mutation") -- the planner only sees category-level summaries (see
        category_summary_prompt()), so it's expected to name a category
        directly rather than guess at tools it's never individually seen.
        Falls back to the seed unchanged if none of it maps to anything known.
        """
        category_names = set(self.all_category_purpose().keys())
        categories: set = set()
        literal_tools: List[str] = []

        for item in seed:
            if item in category_names:
                categories.add(item)
                continue
            literal_tools.append(item)
            cat = self.category_of(item)
            if cat:
                categories.add(cat)

        if not categories:
            return list(dict.fromkeys(literal_tools))

        expanded = list(literal_tools)
        for name in self.tools:
            if self.category_of(name) in categories and name not in expanded:
                expanded.append(name)
        return list(dict.fromkeys(expanded))

    def category_summary_prompt(self) -> str:
        """
        Compact, planner-facing summary: one line per category with its
        purpose and how many currently-registered tools live there --
        deliberately never a per-tool description. This is the piece that
        keeps the planner's prompt size roughly constant regardless of how
        many MCP servers/tools are registered: it always sees a handful of
        category lines, never a growing flat tool list.
        """
        counts: Dict[str, int] = {}
        for name in self.tools:
            cat = self.category_of(name) or "uncategorized"
            counts[cat] = counts.get(cat, 0) + 1

        lines = []
        for cat, purpose in self.all_category_purpose().items():
            if counts.get(cat):
                lines.append(f"- {cat} ({counts[cat]} tools): {purpose}")

        if counts.get("uncategorized"):
            lines.append(
                f"- uncategorized ({counts['uncategorized']} tools): "
                "no declared category -- use read_file/search_code to inspect what these do."
            )

        return "\n".join(lines) if lines else "No tools registered."

    @classmethod
    def build(cls, tools: Dict[str, Callable[..., Any]], mcp_configs: Optional[List[Any]] = None) -> "ToolRegistry":
        """
        Build a registry from an assembled tool map (built-in + MCP, already
        merged and namespaced) plus the MCPServerConfig list that produced
        the MCP half -- used to recover each MCP tool's declared category by
        stripping its "{namespace}__" prefix and looking it up per-server.
        """
        extra_categories: Dict[str, str] = {}
        extra_purpose: Dict[str, str] = {}

        for cfg in mcp_configs or []:
            prefix = f"{cfg.namespace}__"
            for name in tools:
                if not name.startswith(prefix):
                    continue
                base_name = name[len(prefix):]
                cat = cfg.tool_categories.get(base_name) or cfg.category
                if not cat:
                    continue
                extra_categories[name] = cat
                if cfg.category_purpose and cat not in TOOL_CATEGORIES:
                    extra_purpose[cat] = cfg.category_purpose

        return cls(tools=tools, extra_categories=extra_categories, extra_category_purpose=extra_purpose)
