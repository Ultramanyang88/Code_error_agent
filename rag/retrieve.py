from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import re

import numpy as np

from .embedder import CodeEmbedder
from .indexer import RepoIndexer, CodeChunk


class RAGEngine:
    """
    FAISS-based RAG engine for coding agent.

    Main flow:
    1. Build or load FAISS index.
    2. Recall candidates by vector similarity.
    3. Recall candidates by keyword matching.
    4. Merge candidates.
    5. Rerank using simple lexical + vector score.
    """

    def __init__(
        self,
        repo_root: str,
        index_dir: str = ".agent_index",
        embedder: Optional[CodeEmbedder] = None,
        auto_load: bool = True,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.index_dir = index_dir
        self.embedder = embedder or CodeEmbedder()
        self.indexer = RepoIndexer(
            repo_root=str(self.repo_root),
            index_dir=index_dir,
            embedder=self.embedder,
        )

        self.index = None
        self.chunks: List[CodeChunk] = []

        if auto_load:
            self.load_or_build()

    def load_or_build(self, force_rebuild: bool = False) -> None:
        index_path = self.repo_root / self.index_dir / "index.faiss"
        metadata_path = self.repo_root / self.index_dir / "metadata.json"

        if not force_rebuild and index_path.exists() and metadata_path.exists():
            self.index, self.chunks = self.indexer.load()
        else:
            self.index, self.chunks = self.indexer.build(force_rebuild=True)

    def rebuild(self) -> None:
        self.index, self.chunks = self.indexer.build(force_rebuild=True)

    def similarity_search(
        self,
        query: str,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search using FAISS.
        """
        if self.index is None or not self.chunks:
            self.load_or_build()

        query_embedding = self.embedder.embed_query(query)

        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            results.append(
                self._format_result(
                    chunk=chunk,
                    score=float(score),
                    source="vector",
                    extra={"vector_score": float(score)},
                )
            )

        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Lightweight keyword-based recall.

        This is useful for:
        - function names
        - class names
        - import names
        - error messages
        """
        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        scored: List[Tuple[float, CodeChunk, Dict[str, Any]]] = []

        for chunk in self.chunks:
            text = self._chunk_search_text(chunk)
            tokens = self._tokenize(text)

            if not tokens:
                continue

            score_detail = self._keyword_score(query_terms, tokens, text)

            if score_detail["score"] > 0:
                scored.append((score_detail["score"], chunk, score_detail))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []

        for score, chunk, detail in scored[:top_k]:
            results.append(
                self._format_result(
                    chunk=chunk,
                    score=float(score),
                    source="keyword",
                    extra={
                        "keyword_score": float(score),
                        "matched_terms": detail.get("matched_terms", []),
                    },
                )
            )

        return results

    def hybrid_recall(
        self,
        query: str,
        vector_top_k: int = 12,
        keyword_top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid recall = vector search + keyword search.
        """
        vector_results = self.similarity_search(query, top_k=vector_top_k)
        keyword_results = self.keyword_search(query, top_k=keyword_top_k)

        merged = self._merge_results(vector_results, keyword_results)

        return merged

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Simple reranker.

        Later you can replace this with a cross-encoder:
        - BAAI/bge-reranker-base
        - cross-encoder/ms-marco-MiniLM-L-6-v2
        """
        query_terms = set(self._tokenize(query))

        reranked = []

        for item in candidates:
            content = item.get("content", "")
            file_path = item.get("file_path", "")
            symbol_name = item.get("symbol_name") or ""

            text = f"{file_path}\n{symbol_name}\n{content}"
            tokens = set(self._tokenize(text))

            lexical_overlap = 0.0
            if query_terms:
                lexical_overlap = len(query_terms.intersection(tokens)) / len(query_terms)

            vector_score = float(item.get("vector_score", 0.0))
            keyword_score = float(item.get("keyword_score", 0.0))

            path_bonus = self._path_relevance_bonus(query, file_path, symbol_name)

            final_score = (
                0.55 * vector_score
                + 0.30 * min(keyword_score, 1.0)
                + 0.10 * lexical_overlap
                + 0.05 * path_bonus
            )

            new_item = dict(item)
            new_item["rerank_score"] = float(final_score)
            new_item["lexical_overlap"] = float(lexical_overlap)
            new_item["path_bonus"] = float(path_bonus)

            reranked.append(new_item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        vector_top_k: int = 12,
        keyword_top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        End-to-end retrieval.
        """
        candidates = self.hybrid_recall(
            query=query,
            vector_top_k=vector_top_k,
            keyword_top_k=keyword_top_k,
        )

        return self.rerank(query=query, candidates=candidates, top_k=top_k)

    def format_context(
        self,
        results: List[Dict[str, Any]],
        max_chars_per_chunk: int = 1400,
    ) -> str:
        """
        Format retrieval result for LLM context.
        """
        if not results:
            return "No relevant context found."

        blocks = []

        for idx, item in enumerate(results, start=1):
            content = item.get("content", "")

            if len(content) > max_chars_per_chunk:
                content = content[:max_chars_per_chunk] + "\n...[truncated]"

            file_path = item.get("file_path", "unknown")
            start_line = item.get("start_line", "?")
            end_line = item.get("end_line", "?")
            symbol_name = item.get("symbol_name")
            chunk_type = item.get("chunk_type")
            score = item.get("rerank_score", item.get("score", 0.0))

            header = (
                f"[Context {idx}] {file_path}:{start_line}-{end_line}\n"
                f"symbol: {symbol_name}\n"
                f"type: {chunk_type}\n"
                f"score: {score:.4f}"
            )

            blocks.append(
                f"{header}\n```{item.get('language', '')}\n{content}\n```"
            )

        return "\n\n".join(blocks)

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge vector and keyword candidates by chunk_id.
        """
        merged: Dict[str, Dict[str, Any]] = {}

        for item in vector_results:
            chunk_id = item["chunk_id"]
            merged[chunk_id] = dict(item)

        for item in keyword_results:
            chunk_id = item["chunk_id"]

            if chunk_id not in merged:
                merged[chunk_id] = dict(item)
            else:
                existing = merged[chunk_id]
                existing["source"] = "hybrid"
                existing["keyword_score"] = item.get("keyword_score", 0.0)
                existing["matched_terms"] = item.get("matched_terms", [])
                existing["score"] = max(
                    float(existing.get("score", 0.0)),
                    float(item.get("score", 0.0)),
                )

        return list(merged.values())

    def _format_result(
        self,
        chunk: CodeChunk,
        score: float,
        source: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = {
            "chunk_id": chunk.chunk_id,
            "file_path": chunk.file_path,
            "content": chunk.content,
            "chunk_type": chunk.chunk_type,
            "symbol_name": chunk.symbol_name,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "language": chunk.language,
            "score": float(score),
            "source": source,
            "metadata": chunk.metadata,
        }

        if extra:
            data.update(extra)

        return data

    def _chunk_search_text(self, chunk: CodeChunk) -> str:
        return (
            f"{chunk.file_path}\n"
            f"{chunk.symbol_name or ''}\n"
            f"{chunk.chunk_type}\n"
            f"{chunk.language}\n"
            f"{chunk.content}"
        )

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        # Split camelCase / snake_case / normal text roughly.
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+", text.lower())

        tokens = []
        for tok in raw_tokens:
            parts = tok.split("_")
            tokens.extend([p for p in parts if p])

        stopwords = {
            "the", "a", "an", "to", "of", "and", "or", "in", "on",
            "for", "with", "is", "are", "be", "this", "that", "it",
        }

        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _keyword_score(
        self,
        query_terms: List[str],
        doc_tokens: List[str],
        raw_text: str,
    ) -> Dict[str, Any]:
        doc_token_set = set(doc_tokens)
        matched_terms = []

        score = 0.0

        for term in query_terms:
            if term in doc_token_set:
                matched_terms.append(term)

                tf = doc_tokens.count(term)
                score += 1.0 + math.log(1 + tf)

            # Extra bonus for exact raw substring.
            if term in raw_text.lower():
                score += 0.2

        normalized = score / max(1, len(query_terms))

        return {
            "score": normalized,
            "matched_terms": matched_terms,
        }

    def _path_relevance_bonus(
        self,
        query: str,
        file_path: str,
        symbol_name: Optional[str],
    ) -> float:
        query_lower = query.lower()
        file_lower = file_path.lower()
        symbol_lower = (symbol_name or "").lower()

        bonus = 0.0

        important_terms = self._tokenize(query_lower)

        for term in important_terms:
            if term in file_lower:
                bonus += 0.5
            if term in symbol_lower:
                bonus += 0.5

        return min(bonus, 1.0)