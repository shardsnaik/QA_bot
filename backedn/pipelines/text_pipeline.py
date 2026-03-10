"""
RAG Text Pipeline — orchestrates hybrid retrieval (BM25 + vector search).

Ingestion:  text → chunk → embed → store in Milvus + BM25 index
Query:      question → embed → vector search + BM25 → RRF fusion → LLM answer
"""

from __future__ import annotations

import logging
from utils.text_chunker import chunk_text
from utils.embedding_manager import get_embeddings, get_query_embedding
from utils.vector_store import add_documents as vs_add, vector_search
from utils.bm25_retriever import BM25Retriever
from utils.llm_client import generate_answer
from utils.config import TOP_K, BM25_WEIGHT, VECTOR_WEIGHT

logger = logging.getLogger(__name__)

# Module-level BM25 retriever (shared across calls within the same worker process)
_bm25 = BM25Retriever()


# ──────────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────────

def ingest(doc_id: str, text: str) -> dict:
    """
    Ingest a document: chunk → embed → store in Milvus and BM25 index.

    Returns a summary dict with doc_id and chunk count.
    """
    # 1. Chunk
    chunks = chunk_text(text)
    logger.info("Chunked doc '%s' into %d pieces", doc_id, len(chunks))

    # 2. Embed
    embeddings = get_embeddings(chunks)
    logger.info("Generated %d embeddings for doc '%s'", len(embeddings), doc_id)

    # 3. Store in Milvus
    inserted = vs_add(doc_id, chunks, embeddings)
    logger.info("Stored %d chunks in Milvus for doc '%s'", inserted, doc_id)

    # 4. Add to BM25 index
    _bm25.add_documents(doc_id, chunks)

    return {"doc_id": doc_id, "chunks": len(chunks), "stored": inserted}


# ──────────────────────────────────────────────────────────────
# Hybrid Retrieval (BM25 + Vector) with Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for each document = Σ  1 / (k + rank_i)

    We weight each source independently:
     - vector_weight  (default 0.7)
     - bm25_weight    (default 0.3)
    """
    scores: dict[str, float] = {}       # chunk_key → fused score
    chunk_map: dict[str, dict] = {}     # chunk_key → chunk data

    def _key(hit: dict) -> str:
        return f"{hit['doc_id']}::{hit['chunk_index']}"

    # Score vector results
    for rank, hit in enumerate(vector_results, start=1):
        key = _key(hit)
        scores[key] = scores.get(key, 0.0) + VECTOR_WEIGHT / (k + rank)
        chunk_map[key] = hit

    # Score BM25 results
    for rank, hit in enumerate(bm25_results, start=1):
        key = _key(hit)
        scores[key] = scores.get(key, 0.0) + BM25_WEIGHT / (k + rank)
        if key not in chunk_map:
            chunk_map[key] = hit

    # Sort by fused score descending
    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)

    fused: list[dict] = []
    for key in sorted_keys:
        entry = chunk_map[key].copy()
        entry["rrf_score"] = scores[key]
        fused.append(entry)

    return fused


# ──────────────────────────────────────────────────────────────
# Query
# ──────────────────────────────────────────────────────────────

def query(question: str) -> dict:
    """
    Answer a question using hybrid retrieval + Groq LLM.

    Returns
    -------
    dict
        Keys: answer, sources (list of chunk metadata used).
    """
    # 1. Embed the question
    q_embedding = get_query_embedding(question)

    # 2. Vector search (Milvus)
    vector_hits = vector_search(q_embedding, top_k=TOP_K)
    logger.info("Vector search returned %d hits", len(vector_hits))

    # 3. BM25 keyword search
    bm25_hits = _bm25.search(question, top_k=TOP_K)
    logger.info("BM25 search returned %d hits", len(bm25_hits))

    # 4. Fuse with RRF
    fused = _reciprocal_rank_fusion(vector_hits, bm25_hits)
    top_chunks = fused[:TOP_K]

    # 5. Build context string
    if top_chunks:
        context = "\n\n---\n\n".join(
            f"[Source: {c['doc_id']}, chunk {c['chunk_index']}]\n{c['chunk_text']}"
            for c in top_chunks
        )
    else:
        context = "No relevant context found in the uploaded documents."

    # 6. Generate answer via Groq
    answer = generate_answer(context, question)

    return {
        "answer": answer,
        "sources": [
            {
                "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "rrf_score": c.get("rrf_score", 0),
            }
            for c in top_chunks
        ],
    }


def rebuild_bm25_index() -> None:
    """Rebuild BM25 index from Milvus (call on worker startup)."""
    _bm25.rebuild_from_store()
