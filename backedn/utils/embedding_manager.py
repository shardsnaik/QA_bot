"""
Embedding generation using sentence-transformers (free, local).
Uses all-MiniLM-L6-v2 by default — 384-dimensional vectors.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from utils.config import EMBEDDING_MODEL

# Lazy-loaded singleton so the heavy model is only loaded once per process.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Batch-encode a list of text strings into embedding vectors.

    Returns
    -------
    list[list[float]]
        One 384-dim vector per input text.
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def get_query_embedding(text: str) -> list[float]:
    """Encode a single query string."""
    return get_embeddings([text])[0]


def get_embedding_dimension() -> int:
    """Return the dimensionality of the current embedding model."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
