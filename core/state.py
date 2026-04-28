from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
from enum import Enum
import time
import uuid


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_text(self, max_chars: int = 3000) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        text = f"[{status}] Tool: {self.tool_name}\n"

        if self.output:
            text += f"Output:\n{self.output}\n"

        if self.error:
            text += f"Error:\n{self.error}\n"

        if self.metadata:
            text += f"Metadata:\n{self.metadata}\n"

        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"

        return text


@dataclass
class PlanStep:
    step_id: int
    task: str
    expected_output: str = ""
    suggested_tools: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING

    # Planner observability
    planner_notes: str = ""

    # Executor observability
    execution_trace: List[str] = field(default_factory=list)

    result: Optional[str] = None
    error: Optional[str] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    retry_count: int = 0

    @property
    def is_completed(self) -> bool:
        return self.status == StepStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.status == StepStatus.FAILED

    def mark_running(self) -> None:
        self.status = StepStatus.RUNNING

    def mark_completed(self, result: str = "") -> None:
        self.status = StepStatus.COMPLETED
        self.result = result
        self.error = None

    def mark_failed(self, error: str) -> None:
        self.status = StepStatus.FAILED
        self.error = error
        self.retry_count += 1

    def add_trace(self, message: str) -> None:
        self.execution_trace.append(message)


@dataclass
class AgentState:
    input_query: str
    repo_root: str

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    plan: List[PlanStep] = field(default_factory=list)

    history: List[str] = field(default_factory=list)
    tool_history: List[ToolResult] = field(default_factory=list)

    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)

    files_read: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)

    test_results: List[ToolResult] = field(default_factory=list)
    errors_seen: List[str] = field(default_factory=list)

    is_finished: bool = False
    final_answer: Optional[str] = None

    max_steps: int = 20
    max_retries_per_step: int = 2

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.repo_root = str(Path(self.repo_root).resolve())

    def repo_path(self, relative_path: str = "") -> Path:
        """
        Safely resolve a path inside repo root.
        """
        root = Path(self.repo_root).resolve()
        target = (root / relative_path).resolve()

        if not str(target).startswith(str(root)):
            raise ValueError(f"Unsafe path outside repo root: {relative_path}")

        return target

    def add_plan(self, steps: List[PlanStep]) -> None:
        self.plan = steps

    def get_current_step(self) -> Optional[PlanStep]:
        for step in self.plan:
            if step.status in {StepStatus.PENDING, StepStatus.FAILED}:
                if step.retry_count <= self.max_retries_per_step:
                    return step
        return None

    def add_tool_result(self, result: ToolResult) -> None:
        self.tool_history.append(result)

        if result.tool_name == "read_file" and result.success:
            path = result.metadata.get("path")
            if path and path not in self.files_read:
                self.files_read.append(path)

        if result.tool_name in {"apply_patch", "write_file", "replace_in_file"} and result.success:
            changed_files = result.metadata.get("changed_files", [])
            for file_path in changed_files:
                if file_path not in self.files_modified:
                    self.files_modified.append(file_path)

        if result.tool_name in {"run_tests", "run_command"}:
            self.test_results.append(result)

        if not result.success and result.error:
            self.errors_seen.append(result.error)

    def add_history(self, message: str) -> None:
        self.history.append(message)

    def plan_summary(self, verbose: bool = False) -> str:
        if not self.plan:
            return "No plan has been created yet."

        lines: List[str] = []

        for step in self.plan:
            lines.append(f"{step.step_id}. [{step.status.value}] {step.task}")

            if verbose:
                if step.expected_output:
                    lines.append(f"   Expected: {step.expected_output}")
                if step.suggested_tools:
                    lines.append(f"   Suggested tools: {step.suggested_tools}")
                if step.planner_notes:
                    lines.append(f"   Planner notes: {step.planner_notes}")

            if step.error:
                lines.append(f"   Error: {step.error}")

        return "\n".join(lines)

    def recent_tool_summary(self, limit: int = 5) -> str:
        recent = self.tool_history[-limit:]

        if not recent:
            return "No tool calls yet."

        return "\n\n".join(r.to_text(max_chars=1200) for r in recent)

    def retrieved_context_text(
        self,
        max_chunks: int = 6,
        max_chars_per_chunk: int = 1200,
    ) -> str:
        if not self.retrieved_context:
            return "No retrieved context."

        blocks: List[str] = []

        for item in self.retrieved_context[:max_chunks]:
            path = item.get("file_path", "unknown")
            start = item.get("start_line", "?")
            end = item.get("end_line", "?")
            content = item.get("content", "")

            if len(content) > max_chars_per_chunk:
                content = content[:max_chars_per_chunk] + "\n...[truncated]"

            blocks.append(
                f"File: {path}:{start}-{end}\n"
                f"```python\n{content}\n```"
            )

        return "\n\n".join(blocks)

    def should_stop(self) -> bool:
        if self.is_finished:
            return True

        completed_or_failed = [
            s
            for s in self.plan
            if s.status in {
                StepStatus.COMPLETED,
                StepStatus.FAILED,
                StepStatus.SKIPPED,
            }
        ]

        if self.plan and len(completed_or_failed) == len(self.plan):
            return True

        if len(self.tool_history) >= self.max_steps:
            return True

        return False