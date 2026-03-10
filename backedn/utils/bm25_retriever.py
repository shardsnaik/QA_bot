"""
BM25 keyword-based retriever using rank-bm25.
Maintains an in-memory index that can be rebuilt from the Milvus store.
"""

from __future__ import annotations

import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Keyword-based retrieval using BM25 (Okapi variant).

    The index is held in memory and can be rebuilt from vector-store data
    on worker startup via ``rebuild_from_store()``.
    """

    def __init__(self) -> None:
        self._corpus_chunks: list[dict] = []   # [{doc_id, chunk_index, chunk_text}]
        self._tokenised: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    # ── Index management ─────────────────────────

    def add_documents(self, doc_id: str, chunks: list[str]) -> None:
        """Add new chunks to the BM25 index (incremental)."""
        for i, chunk in enumerate(chunks):
            self._corpus_chunks.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_text": chunk,
            })
            self._tokenised.append(self._tokenise(chunk))

        self._rebuild_index()
        logger.info("BM25 index updated — total chunks: %d", len(self._corpus_chunks))

    def rebuild_from_store(self) -> None:
        """
        Rebuild the entire BM25 index from all chunks stored in Milvus.
        Call this on worker startup to hydrate the keyword index.
        """
        from utils.vector_store import get_all_chunks  # avoid circular imports

        all_chunks = get_all_chunks()
        self._corpus_chunks = all_chunks
        self._tokenised = [self._tokenise(c["chunk_text"]) for c in all_chunks]
        self._rebuild_index()
        logger.info("BM25 index rebuilt from store — total chunks: %d", len(self._corpus_chunks))

    # ── Search ───────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return the top-k chunks ranked by BM25 score.

        Returns
        -------
        list[dict]
            Each dict has: doc_id, chunk_index, chunk_text, score.
        """
        if self._bm25 is None or len(self._corpus_chunks) == 0:
            return []

        tokens = self._tokenise(query)
        scores = self._bm25.get_scores(tokens)

        # Pair scores with chunks, sort descending
        scored = sorted(
            zip(scores, self._corpus_chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        results: list[dict] = []
        for score, chunk in scored[:top_k]:
            results.append({
                "doc_id": chunk["doc_id"],
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
                "score": float(score),
            })
        return results

    # ── Internal helpers ─────────────────────────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Simple whitespace + lower-case tokeniser."""
        return text.lower().split()

    def _rebuild_index(self) -> None:
        if self._tokenised:
            self._bm25 = BM25Okapi(self._tokenised)
        else:
            self._bm25 = None
