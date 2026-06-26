from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import ast
import hashlib
import json
import os

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss is not installed. Run: pip install faiss-cpu"
    ) from exc

from .embedder import CodeEmbedder


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".DS_Store",
}

SUPPORTED_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".sql",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".hpp",
    ".c",
    ".h",
}


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    content: str
    chunk_type: str
    symbol_name: Optional[str]
    start_line: int
    end_line: int
    language: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeChunk":
        return cls(**data)


class RepoIndexer:
    """
    Build and persist a FAISS index for a code repository.

    Files generated:
    - .agent_index/index.faiss
    - .agent_index/metadata.json
    """

    def __init__(
        self,
        repo_root: str,
        index_dir: str = ".agent_index",
        embedder: Optional[CodeEmbedder] = None,
        max_file_size_kb: int = 512,
        text_chunk_lines: int = 80,
        text_chunk_overlap: int = 15,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.index_dir = self.repo_root / index_dir
        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.json"

        self.embedder = embedder or CodeEmbedder()
        self.max_file_size_kb = max_file_size_kb
        self.text_chunk_lines = text_chunk_lines
        self.text_chunk_overlap = text_chunk_overlap

        self.index = None
        self.chunks: List[CodeChunk] = []

    def build(self, force_rebuild: bool = True) -> Tuple[Any, List[CodeChunk]]:
        """
        Build FAISS index from repository files.
        """
        if not force_rebuild and self.index_path.exists() and self.metadata_path.exists():
            self.load()
            return self.index, self.chunks

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.chunks = self._collect_chunks()

        if not self.chunks:
            raise ValueError("No chunks found. Check repo_root and supported file extensions.")

        texts = [self._chunk_to_embedding_text(chunk) for chunk in self.chunks]
        embeddings = self.embedder.embed_texts(texts)

        dim = embeddings.shape[1]

        # Since embeddings are normalized, inner product is equivalent to cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.save()

        return self.index, self.chunks

    def save(self) -> None:
        if self.index is None:
            raise ValueError("No FAISS index to save.")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))

        metadata = {
            "repo_root": str(self.repo_root),
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.embedding_dim,
            "num_chunks": len(self.chunks),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self) -> Tuple[Any, List[CodeChunk]]:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))

        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.chunks = [CodeChunk.from_dict(item) for item in data["chunks"]]

        return self.index, self.chunks

    def _collect_chunks(self) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []

        chunks.extend(self._inject_tool_spec_chunks())
        chunks.extend(self._inject_skill_chunks())

        for file_path in self.repo_root.rglob("*"):
            if not file_path.is_file():
                continue

            rel_path = file_path.relative_to(self.repo_root)

            if self._should_ignore(rel_path):
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            if self._too_large(file_path):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if not content.strip():
                continue

            if file_path.suffix.lower() == ".py":
                file_chunks = self._chunk_python_file(rel_path, content)
            else:
                file_chunks = self._chunk_text_file(rel_path, content)

            chunks.extend(file_chunks)

        return chunks

    def _inject_tool_spec_chunks(self) -> List[CodeChunk]:
        """Create one synthetic chunk per registered tool, from TOOL_SPECS."""
        chunks: List[CodeChunk] = []
        try:
            import sys as _sys
            _sys.path.insert(0, str(self.repo_root))
            from tools.specs import TOOL_SPECS
        except Exception:
            return chunks

        for tool_name, spec in TOOL_SPECS.items():
            desc = spec.get("description", "")
            when_to_use = "\n".join(f"- {u}" for u in spec.get("when_to_use", []))
            params = spec.get("parameters", {})
            param_lines = []
            for p_name, p_meta in params.items():
                req = " [required]" if p_meta.get("required") else ""
                param_lines.append(f"  {p_name}: {p_meta.get('description', '')}{req}")
            params_text = "\n".join(param_lines) or "  (no parameters)"

            content = (
                f"Tool: {tool_name}\n"
                f"Description: {desc}\n"
                f"When to use:\n{when_to_use}\n"
                f"Parameters:\n{params_text}\n"
                f"Output: {spec.get('output', '')}"
            )

            chunks.append(self._make_chunk(
                file_path=f"__knowledge__/tools/{tool_name}",
                content=content,
                chunk_type="tool_summary",
                symbol_name=tool_name,
                start_line=1,
                end_line=content.count("\n") + 1,
                language="text",
                metadata={"kind": "tool_summary", "tool_name": tool_name},
            ))

        return chunks

    def _inject_skill_chunks(self) -> List[CodeChunk]:
        """Create one synthetic chunk per skill .md file."""
        chunks: List[CodeChunk] = []
        skills_dir = self.repo_root / "skills"
        if not skills_dir.exists():
            return chunks

        import re as _re
        for path in sorted(skills_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            match = _re.match(r"^---\n(.*?)\n---\n(.*)", text, _re.DOTALL)
            if not match:
                continue
            frontmatter, body = match.group(1), match.group(2).strip()

            name_m = _re.search(r"^name:\s*(.+)$", frontmatter, _re.MULTILINE)
            kw_m = _re.search(r"^trigger_keywords:\s*\[(.+)\]$", frontmatter, _re.MULTILINE)
            sum_m = _re.search(r"^summary:\s*(.+)$", frontmatter, _re.MULTILINE)
            if not (name_m and sum_m):
                continue

            name = name_m.group(1).strip()
            summary = sum_m.group(1).strip()
            keywords = kw_m.group(1) if kw_m else ""

            content = (
                f"Skill: {name}\n"
                f"Trigger keywords: {keywords}\n"
                f"Summary: {summary}\n\n"
                f"Procedure:\n{body}"
            )

            chunks.append(self._make_chunk(
                file_path=f"__knowledge__/skills/{path.stem}",
                content=content,
                chunk_type="skill_summary",
                symbol_name=name,
                start_line=1,
                end_line=content.count("\n") + 1,
                language="text",
                metadata={"kind": "skill_summary", "skill_name": name},
            ))

        return chunks

    def _chunk_python_file(self, rel_path: Path, content: str) -> List[CodeChunk]:
        """
        Chunk Python file by class/function when possible.
        Also include import/module-level chunk.
        """
        lines = content.splitlines()
        chunks: List[CodeChunk] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._chunk_text_file(rel_path, content, language="python")

        # Module import chunk.
        import_lines = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", start)
                import_lines.extend(lines[start - 1:end])

        if import_lines:
            import_content = "\n".join(import_lines)
            chunks.append(
                self._make_chunk(
                    file_path=str(rel_path),
                    content=import_content,
                    chunk_type="imports",
                    symbol_name=None,
                    start_line=1,
                    end_line=max(1, len(import_lines)),
                    language="python",
                    metadata={"kind": "imports"},
                )
            )

        symbol_nodes = []

        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol_nodes.append(node)

        for node in symbol_nodes:
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)

            symbol_content = "\n".join(lines[start - 1:end])
            symbol_name = getattr(node, "name", None)

            if isinstance(node, ast.ClassDef):
                chunk_type = "class"
            elif isinstance(node, ast.AsyncFunctionDef):
                chunk_type = "async_function"
            else:
                chunk_type = "function"

            chunks.append(
                self._make_chunk(
                    file_path=str(rel_path),
                    content=symbol_content,
                    chunk_type=chunk_type,
                    symbol_name=symbol_name,
                    start_line=start,
                    end_line=end,
                    language="python",
                    metadata={
                        "kind": chunk_type,
                        "symbol_name": symbol_name,
                    },
                )
            )

            # For class methods, also create method-level chunks.
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start = getattr(child, "lineno", start)
                        m_end = getattr(child, "end_lineno", m_start)
                        method_content = "\n".join(lines[m_start - 1:m_end])
                        method_name = f"{node.name}.{child.name}"

                        chunks.append(
                            self._make_chunk(
                                file_path=str(rel_path),
                                content=method_content,
                                chunk_type="method",
                                symbol_name=method_name,
                                start_line=m_start,
                                end_line=m_end,
                                language="python",
                                metadata={
                                    "kind": "method",
                                    "class_name": node.name,
                                    "method_name": child.name,
                                    "symbol_name": method_name,
                                },
                            )
                        )

        # Fallback: if there are no functions/classes, chunk whole file.
        if not chunks:
            chunks = self._chunk_text_file(rel_path, content, language="python")

        return chunks

    def _chunk_text_file(
        self,
        rel_path: Path,
        content: str,
        language: Optional[str] = None,
    ) -> List[CodeChunk]:
        lines = content.splitlines()

        if language is None:
            language = self._detect_language(rel_path)

        chunks: List[CodeChunk] = []

        step = max(1, self.text_chunk_lines - self.text_chunk_overlap)

        for start_idx in range(0, len(lines), step):
            end_idx = min(len(lines), start_idx + self.text_chunk_lines)

            selected = lines[start_idx:end_idx]
            selected_content = "\n".join(selected)

            if not selected_content.strip():
                continue

            chunks.append(
                self._make_chunk(
                    file_path=str(rel_path),
                    content=selected_content,
                    chunk_type="text_block",
                    symbol_name=None,
                    start_line=start_idx + 1,
                    end_line=end_idx,
                    language=language,
                    metadata={"kind": "text_block"},
                )
            )

            if end_idx >= len(lines):
                break

        return chunks

    def _make_chunk(
        self,
        file_path: str,
        content: str,
        chunk_type: str,
        symbol_name: Optional[str],
        start_line: int,
        end_line: int,
        language: str,
        metadata: Dict[str, Any],
    ) -> CodeChunk:
        raw = f"{file_path}:{start_line}:{end_line}:{content}"
        chunk_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=chunk_type,
            symbol_name=symbol_name,
            start_line=start_line,
            end_line=end_line,
            language=language,
            metadata=metadata,
        )

    def _chunk_to_embedding_text(self, chunk: CodeChunk) -> str:
        header = (
            f"file_path: {chunk.file_path}\n"
            f"chunk_type: {chunk.chunk_type}\n"
            f"symbol_name: {chunk.symbol_name}\n"
            f"language: {chunk.language}\n"
            f"lines: {chunk.start_line}-{chunk.end_line}\n"
        )
        return header + "\n" + chunk.content

    def _detect_language(self, rel_path: Path) -> str:
        ext = rel_path.suffix.lower()

        mapping = {
            ".py": "python",
            ".md": "markdown",
            ".txt": "text",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".ini": "ini",
            ".cfg": "config",
            ".sh": "shell",
            ".sql": "sql",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript-react",
            ".jsx": "javascript-react",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".c": "c",
            ".h": "c-header",
        }

        return mapping.get(ext, "text")

    def _should_ignore(self, rel_path: Path) -> bool:
        parts = set(rel_path.parts)

        if parts.intersection(IGNORED_DIRS):
            return True

        if rel_path.parts and rel_path.parts[0] == ".agent_index":
            return True

        return False

    def _too_large(self, file_path: Path) -> bool:
        try:
            size_kb = file_path.stat().st_size / 1024
            return size_kb > self.max_file_size_kb
        except OSError:
            return True