from __future__ import annotations

from typing import List, Union
import numpy as np


class CodeEmbedder:
    """
    Embedding wrapper for code/document chunks.

    Default model:
    - sentence-transformers/all-MiniLM-L6-v2

    Later you can switch to:
    - BAAI/bge-small-en-v1.5
    - mixedbread-ai/mxbai-embed-large-v1
    - nomic-ai/nomic-embed-text-v1.5
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.model = None
        self.embedding_dim = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(self.model_name)

        # Detect dimension using a small dummy embedding.
        dummy = self.model.encode(
            ["dimension check"],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        self.embedding_dim = int(dummy.shape[1])

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple text chunks.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype="float32")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed one query.
        """
        if not query or not query.strip():
            query = "empty query"

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        return embedding.astype("float32")

    def embed(self, text_or_texts: Union[str, List[str]]) -> np.ndarray:
        """
        Convenience method.
        """
        if isinstance(text_or_texts, str):
            return self.embed_query(text_or_texts)

        return self.embed_texts(text_or_texts)