"""
Módulo de NLP y Embeddings Semánticos usando SentenceTransformers.
"""
from __future__ import annotations
from functools import lru_cache
from typing import List, Sequence
import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Carga y almacena en caché el modelo SentenceTransformer ('all-MiniLM-L6-v2')."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


@lru_cache(maxsize=2048)
def embed_text(text: str) -> np.ndarray:
    """Genera el embedding vectorial normalizado para una sinopsis."""
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True)


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Genera embeddings vectoriales normalizados por lotes (batch) para una lista de textos."""
    model = get_embedding_model()
    return np.array(model.encode(list(texts), normalize_embeddings=True))
