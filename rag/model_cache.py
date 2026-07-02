from __future__ import annotations
from typing import Optional
from .embedder import CodeEmbedder

_embedder: Optional[CodeEmbedder] = None
_cross_encoder = None
_cross_encoder_load_failed = False

def get_shared_embedder() -> CodeEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = CodeEmbedder()
    return _embedder

def get_shared_cross_encoder():
    global _cross_encoder, _cross_encoder_load_failed
    if _cross_encoder is None and not _cross_encoder_load_failed:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _cross_encoder_load_failed = True
    return _cross_encoder