from __future__ import annotations

"""
Unified retrieval across sources that otherwise don't share a ranking
pipeline.

Repo code and the injected knowledge-base chunks (tool/skill descriptions,
see indexer.py's _inject_tool_spec_chunks/_inject_skill_chunks) are already
merged into one FAISS index and share RAGEngine's hybrid recall + rerank.
AgentMemory's long-term insights/preferences are a separate store with no
vector index of their own. FederatedRetriever recalls a batch of candidates
from each source, tags them with source_type, and reranks the *combined*
pool through RAGEngine's existing reranker (cross-encoder if available,
lexical fallback otherwise) -- so the most relevant result wins regardless
of which source it came from, instead of the caller having to manually
interleave two separately-ranked lists.

Not wired into any tool yet: tools/tools.py's retrieve_context() still calls
RAGEngine.retrieve() directly (code + knowledge base only). Routing it
through FederatedRetriever.retrieve() instead is a one-line swap once memory
recall has been validated in practice; a future "database retrieve" source
(e.g. structured run history) plugs in the same way -- recall a batch,
tag source_type="database", add it to the merged pool before rerank.
"""

from typing import Any, Dict, List, Optional

from .retrieve import RAGEngine


class FederatedRetriever:
    def __init__(self, rag_engine: RAGEngine, memory: Optional[Any] = None):
        self.rag_engine = rag_engine
        self.memory = memory  # an AgentMemory instance, or None to skip that source

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        code_vector_top_k: int = 12,
        code_keyword_top_k: int = 20,
        memory_top_k: int = 5,
        min_score: Optional[float] = -4.0,
    ) -> List[Dict[str, Any]]:
        # 1. recall: one batch per source, independently
        code_candidates = self.rag_engine.hybrid_recall(
            query=query,
            vector_top_k=code_vector_top_k,
            keyword_top_k=code_keyword_top_k,
        )
        for c in code_candidates:
            c.setdefault("source_type", "repo")

        memory_candidates: List[Dict[str, Any]] = []
        if self.memory is not None:
            memory_candidates = self.memory.recall_candidates(query, top_k=memory_top_k)
            for c in memory_candidates:
                c["source_type"] = "memory"

        candidates = code_candidates + memory_candidates
        if not candidates:
            return []

        # 2. rerank: one shared pass over the merged pool, so relevance is
        # compared on the same scale regardless of source.
        reranked = self.rag_engine.rerank(query=query, candidates=candidates, top_k=top_k)
        return self.rag_engine.apply_relevance_floor(reranked, min_score)

    def format_context(self, results: List[Dict[str, Any]], max_chars_per_chunk: int = 1400) -> str:
        """Delegates to RAGEngine's formatter -- same block layout regardless of source_type."""
        return self.rag_engine.format_context(results, max_chars_per_chunk=max_chars_per_chunk)
