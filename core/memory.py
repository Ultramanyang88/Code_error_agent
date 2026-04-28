from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from .state import ToolResult
# short term will in tool results
# long term will store first as insight then as vector db

@dataclass
class MemoryItem:
    content: str
    memory_type: str = "general"
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class AgentMemory:
    """
    Memory module for the coding agent.

    Short-term memory:
    - Recent tool results
    - Current errors
    - Recently read files
    - Test output

    Long-term memory:
    - Repo insights
    - Architecture summaries
    - Previously discovered bugs
    - Useful implementation facts
    """

    def __init__(self, short_term_limit: int = 12):
        self.short_term_limit = short_term_limit
        self.short_term: List[ToolResult] = []
        self.long_term: List[MemoryItem] = []

    def add_tool_result(self, result: ToolResult) -> None:
        self.short_term.append(result)

        if len(self.short_term) > self.short_term_limit:
            self.short_term = self.short_term[-self.short_term_limit:]

    def add_insight(
        self,
        insight: str,
        memory_type: str = "repo_insight",
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a reusable discovery.

        Example:
        - "Executor imports from tool, but the actual file is tools.py."
        - "AgentState stores files_modified and test_results."
        """
        item = MemoryItem(
            content=insight,
            memory_type=memory_type,
            source=source,
            metadata=metadata or {},
        )
        self.long_term.append(item)

    def summarize_short_term(self, max_items: int = 8, max_chars: int = 4000) -> str:
        recent = self.short_term[-max_items:]

        if not recent:
            return "No recent tool results."

        blocks = []
        for item in recent:
            blocks.append(item.to_text(max_chars=800))

        text = "\n\n".join(blocks)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text

    def summarize_long_term(self, max_items: int = 8, max_chars: int = 4000) -> str:
        if not self.long_term:
            return "No long-term memory yet."

        recent = self.long_term[-max_items:]
        lines = []

        for item in recent:
            source = f" Source: {item.source}." if item.source else ""
            lines.append(f"- [{item.memory_type}] {item.content}{source}")

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text

    def get_context(self) -> str:
        return (
            "Short-term memory:\n"
            f"{self.summarize_short_term()}\n\n"
            "Long-term memory:\n"
            f"{self.summarize_long_term()}"
        )

    def extract_insight_from_tool_result(self, result: ToolResult) -> Optional[str]:
        """
        A simple rule-based insight extractor.
        Later, this can be replaced by an LLM summarizer.
        """
        if not result.success and result.error:
            return f"Tool {result.tool_name} failed with error: {result.error}"

        if result.tool_name == "read_file" and result.success:
            path = result.metadata.get("path")
            if path:
                return f"Read file {path}; it may be relevant to the current task."

        if result.tool_name == "run_tests":
            if result.success:
                return "Tests passed after the latest execution."
            return "Tests failed; inspect traceback and relevant files before editing again."

        return None

    def update_from_tool_result(self, result: ToolResult) -> None:
        self.add_tool_result(result)

        insight = self.extract_insight_from_tool_result(result)
        if insight:
            self.add_insight(
                insight=insight,
                memory_type="tool_observation",
                source=result.tool_name,
                metadata=result.metadata,
            )