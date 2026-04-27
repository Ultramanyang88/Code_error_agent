from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    tool_name: str
    args: Dict[str, Any]
    output: Any


@dataclass
class AgentState:
    repo_root: str
    validation_cmd: Optional[str] = None
    last_validation_passed: bool = False
    last_validation_output: str = ""
    edited_files: List[str] = field(default_factory=list)
    step_count: int = 0
    history: List[ToolResult] = field(default_factory=list)