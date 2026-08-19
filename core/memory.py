from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time, json
from pathlib import Path
import hashlib

from .state import ToolResult
# short term will in tool results
# long term will store first as insight then as vector db

@dataclass
class MemoryItem:
    content: str
    memory_type: str = "general"  # general | repo_insight | tool_observation | preference
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active | stale | superseded
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


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
    @staticmethod
    def namespace_for(repo_root: str, session_id: Optional[str] = None):
        if session_id:
            return session_id
        return hashlib.sha1(str(Path(repo_root).resolve()).encode()).hexdigest()[:12]

    def __init__(self, short_term_limit: int = 12, persist_dir: Optional[str] = None):
        self.short_term_limit = short_term_limit
        self.short_term: List[ToolResult] = []
        self.long_term: List[MemoryItem] = []
        self._persist_path = Path(persist_dir) / "memory.jsonl" if persist_dir else None
        if self._persist_path:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return

        raw_items: List[MemoryItem] = []
        for line in self._persist_path.read_text().splitlines():
            try:
                d = json.loads(line)
                created_at = d.get("created_at", time.time())
                raw_items.append(MemoryItem(
                    content=d["content"],
                    memory_type=d.get("memory_type", "general"),
                    source=d.get("source"),
                    metadata=d.get("metadata", {}),
                    status=d.get("status", "active"),
                    created_at=created_at,
                    updated_at=d.get("updated_at", created_at),
                ))
            except Exception:
                continue

        # The JSONL file is an append-only log -- it can't be edited in
        # place, so remember_preference() re-appends a new line on every
        # update instead of rewriting the old one. Resolve "preference" items
        # (identified by metadata["key"]) down to just their latest version;
        # every other memory_type is kept as-is (append-only insight log).
        latest_preference: Dict[str, MemoryItem] = {}
        resolved: List[MemoryItem] = []
        for item in raw_items:
            key = item.metadata.get("key") if item.memory_type == "preference" else None
            if key is None:
                resolved.append(item)
                continue
            existing = latest_preference.get(key)
            if existing is None or item.updated_at >= existing.updated_at:
                latest_preference[key] = item

        resolved.extend(latest_preference.values())
        self.long_term = resolved

    def _persist_insight(self, item: MemoryItem) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._persist_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "content": item.content,
                "memory_type": item.memory_type,
                "source": item.source,
                "metadata": item.metadata,
                "status": item.status,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }) + "\n")

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
        self._persist_insight(item)

    def remember_preference(
        self,
        key: str,
        value: str,
        source: Optional[str] = None,
    ) -> MemoryItem:
        """
        Upsert a user/task preference (e.g. "preferred_test_command",
        "code_style"), keyed by `key` so repeated observations converge on
        the latest value instead of piling up as separate, possibly
        contradictory insights every time the same preference is re-observed.

        Safe to call every time -- an existing entry for `key` is updated in
        place (in memory) and re-persisted; a new one is created otherwise.
        """
        now = time.time()

        for item in self.long_term:
            if item.memory_type == "preference" and item.metadata.get("key") == key:
                item.content = value
                item.updated_at = now
                item.metadata["update_count"] = item.metadata.get("update_count", 1) + 1
                self._persist_insight(item)  # append-only log; _load_from_disk resolves to latest
                return item

        item = MemoryItem(
            content=value,
            memory_type="preference",
            source=source,
            metadata={"key": key, "update_count": 1},
            created_at=now,
            updated_at=now,
        )
        self.long_term.append(item)
        self._persist_insight(item)
        return item

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 3,
        memory_types: Optional[List[str]] = None,
    ) -> List[MemoryItem]:
        """Keyword-based relevance filter over active long-term memory."""
        query_lower = query.lower()
        scored = []
        for item in self.long_term:
            if item.status != "active":
                continue
            if memory_types and item.memory_type not in memory_types:
                continue
            score = sum(1 for word in query_lower.split() if word in item.content.lower())
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def recall_candidates(self, query: str, top_k: int = 5, embedder: Any = None) -> List[Dict[str, Any]]:
        """
        Vector recall over active long-term memory, shaped like a RAGEngine
        result dict (chunk_id/content/score/...) so it can be merged into the
        same recall-then-rerank pipeline as code/knowledge-base results --
        see rag/federated.py's FederatedRetriever.

        Reuses the embedding model RAG already loads (rag.model_cache's
        shared singleton) rather than a separate one, and does a brute-force
        cosine scan rather than a FAISS index: long-term memory is typically
        a few dozen to a few hundred items, so this is fast without needing
        its own index to build/persist/invalidate.

        Import of rag.model_cache is deferred to call time (not module load
        time) so constructing an AgentMemory doesn't force-load the embedding
        model for callers that never use this method.
        """
        active = [item for item in self.long_term if item.status == "active"]
        if not active or not query or not query.strip():
            return []

        if embedder is None:
            from rag.model_cache import get_shared_embedder
            embedder = get_shared_embedder()

        doc_embeddings = embedder.embed_texts([item.content for item in active])
        query_embedding = embedder.embed_query(query)[0]
        scores = doc_embeddings @ query_embedding  # both normalized -> cosine similarity

        ranked = sorted(zip(scores.tolist(), active), key=lambda x: x[0], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for score, item in ranked:
            key = item.metadata.get("key") if item.metadata else None
            results.append({
                "chunk_id": f"memory:{item.memory_type}:{key or id(item)}",
                "file_path": f"__memory__/{item.memory_type}",
                "content": item.content,
                "chunk_type": item.memory_type,
                "symbol_name": key,
                "start_line": 1,
                "end_line": 1,
                "language": "text",
                "score": float(score),
                "vector_score": float(score),
                "keyword_score": 0.0,
                "source": "memory",
                "metadata": item.metadata,
            })
        return results
    
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