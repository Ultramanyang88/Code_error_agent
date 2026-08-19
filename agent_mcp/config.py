"""
Config-driven MCP server list (see mcp_servers.json.example for the full
shape -- a list of {command, args, ...} entries, one per server).

Resolution order: explicit `path` argument > $MCP_SERVERS_CONFIG env var >
./mcp_servers.json > the one built-in filesystem server (so an agent with no
config file behaves exactly as it did before this was configurable).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_PATH_ENV = "MCP_SERVERS_CONFIG"
DEFAULT_CONFIG_FILENAME = "mcp_servers.json"

# Preserves current behavior when no config file exists: one filesystem
# server, scoped to the repo being worked on.
_BUILTIN_DEFAULT: List[Dict[str, Any]] = [
    {
        "name": "mcp_fs",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "namespace": "mcp_fs",
        "enabled": True,
        "cwd_from_repo_root": True,
    }
]


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    namespace: str = ""
    enabled: bool = True
    cwd_from_repo_root: bool = False  # run the subprocess with cwd = the repo being worked on
    allowed_tools: Optional[List[str]] = None   # unqualified tool names; None = allow all
    denied_tools: List[str] = field(default_factory=list)  # regex patterns, always applied

    # Routes this server's tools through ToolRegistry's category system
    # (tools/registry.py) like built-in tools. category_purpose is only
    # needed for a category not already in tools/specs.py's TOOL_CATEGORIES.
    category: Optional[str] = None
    tool_categories: Dict[str, str] = field(default_factory=dict)
    category_purpose: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.namespace:
            self.namespace = self.name

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCPServerConfig":
        if "name" not in d or "command" not in d:
            raise ValueError(f"MCP server config entry missing required 'name'/'command': {d}")
        return cls(
            name=d["name"],
            command=d["command"],
            args=list(d.get("args", [])),
            namespace=d.get("namespace", d["name"]),
            enabled=bool(d.get("enabled", True)),
            cwd_from_repo_root=bool(d.get("cwd_from_repo_root", False)),
            allowed_tools=d.get("allowed_tools"),
            denied_tools=list(d.get("denied_tools", [])),
            category=d.get("category"),
            tool_categories=dict(d.get("tool_categories", {})),
            category_purpose=d.get("category_purpose"),
        )


def load_mcp_server_configs(path: Optional[str] = None) -> List[MCPServerConfig]:
    candidate = path or os.environ.get(DEFAULT_CONFIG_PATH_ENV)
    if not candidate:
        default_path = Path(DEFAULT_CONFIG_FILENAME)
        candidate = str(default_path) if default_path.exists() else None

    if candidate:
        if not Path(candidate).exists():
            raise FileNotFoundError(f"MCP server config not found: {candidate}")
        try:
            data = json.loads(Path(candidate).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse MCP server config {candidate}: {exc}") from exc
    else:
        data = _BUILTIN_DEFAULT

    if not isinstance(data, list):
        raise ValueError(f"MCP server config must be a JSON array of server entries, got {type(data).__name__}")

    return [MCPServerConfig.from_dict(entry) for entry in data]


def filter_tool_map(tool_map: Dict[str, Any], config: MCPServerConfig) -> Dict[str, Any]:
    """
    Apply one server's allow/deny policy to its already-namespaced tool map
    (as returned by MCPToolClient.get_tool_map(), e.g. {"mcp_fs__read_file": fn}).

    allowed_tools=None means "no allow-list -- everything passes unless denied".
    denied_tools are regex patterns matched against both the unqualified name
    and the namespaced name; always applied, even with no allow-list, so a
    server that's otherwise fully trusted can still have specific dangerous
    tools blocked (e.g. denied_tools=["delete_.*"]).
    """
    prefix = f"{config.namespace}__"
    filtered: Dict[str, Any] = {}

    for namespaced_name, fn in tool_map.items():
        base_name = namespaced_name[len(prefix):] if namespaced_name.startswith(prefix) else namespaced_name

        if config.allowed_tools is not None and base_name not in config.allowed_tools:
            continue

        if any(re.search(p, base_name) or re.search(p, namespaced_name) for p in config.denied_tools):
            continue

        filtered[namespaced_name] = fn

    return filtered
