"""
Pinecone vector store wrapper for cloud-based vector storage and retrieval.
Handles index creation, document upsertion, and approximate nearest-neighbour search.
"""

from __future__ import annotations

import logging
from pinecone import Pinecone, ServerlessSpec
from utils.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
)

logger = logging.getLogger(__name__)

# ── Embedding dimension (all-MiniLM-L6-v2 = 384) ────────────
EMBEDDING_DIM = 384

_index = None


# ──────────────────────────────────────────────────────────────
# Connection helpers
# ──────────────────────────────────────────────────────────────

def _get_client() -> Pinecone:
    """Create a Pinecone client."""
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index():
    """
    Return the Pinecone index, creating it if it doesn't exist.
    Uses cosine similarity on a serverless spec.
    """
    global _index
    if _index is not None:
        return _index

    pc = _get_client()

    # Create index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        logger.info(
            "Created Pinecone index '%s' (dim=%d, metric=cosine)",
            PINECONE_INDEX_NAME, EMBEDDING_DIM,
        )

    _index = pc.Index(PINECONE_INDEX_NAME)
    logger.info("Connected to Pinecone index '%s'", PINECONE_INDEX_NAME)
    return _index


# ──────────────────────────────────────────────────────────────
# CRUD operations
# ──────────────────────────────────────────────────────────────

def add_documents(
    doc_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    """
    Upsert document chunks with their embeddings into Pinecone.

    Each vector ID is formatted as ``{doc_id}::chunk_{i}`` to allow
    easy filtering and avoid collisions across documents.

    Returns the number of upserted vectors.
    """
    index = ensure_index()

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{doc_id}::chunk_{i}",
            "values": embedding,
            "metadata": {
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_text": chunk,
            },
        })

    # Pinecone recommends batches of 100
    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        index.upsert(vectors=batch)

    logger.info("Upserted %d chunks for doc_id='%s'", len(chunks), doc_id)
    return len(chunks)


def vector_search(
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """
    Perform ANN search on the Pinecone index.

    Returns
    -------
    list[dict]
        Each dict has keys: doc_id, chunk_index, chunk_text, score.
    """
    index = ensure_index()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    hits: list[dict] = []
    for match in results.matches:
        meta = match.metadata or {}
        hits.append({
            "doc_id": meta.get("doc_id", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_text": meta.get("chunk_text", ""),
            "score": match.score,
        })
    return hits


def get_all_chunks() -> list[dict]:
    """
    Retrieve all stored chunks (used to rebuild the BM25 index).

    Uses Pinecone's list + fetch to paginate through all vectors.

    Returns
    -------
    list[dict]
        Each dict has: doc_id, chunk_index, chunk_text.
    """
    index = ensure_index()

    all_chunks: list[dict] = []

    # Paginate through all vector IDs
    for ids_batch in index.list():
        if not ids_batch:
            break
        fetch_response = index.fetch(ids=ids_batch)
        for vec_id, vec_data in fetch_response.vectors.items():
            meta = vec_data.metadata or {}
            all_chunks.append({
                "doc_id": meta.get("doc_id", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "chunk_text": meta.get("chunk_text", ""),
            })

    return all_chunks
