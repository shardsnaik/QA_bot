"""
Celery text worker — bridges the FastAPI route to the RAG pipeline.

Handles two flows:
  • Ingest: file upload → decode text → pipeline.ingest()
  • Query:  chat message → pipeline.query() → return answer
"""

from __future__ import annotations

import logging
from celery import current_app as celery_app

from pipelines.text_pipeline import ingest, query, rebuild_bm25_index
from utils.pdf_extractor import extract_text_from_pdf

logger = logging.getLogger(__name__)


# Rebuild BM25 index on worker startup so keyword search works immediately
@celery_app.on_after_finalize.connect
def _on_startup(sender, **kwargs):
    try:
        rebuild_bm25_index()
        logger.info("BM25 index rebuilt on worker startup")
    except Exception as exc:
        logger.warning("Could not rebuild BM25 index on startup: %s", exc)


@celery_app.task(name="workers.text_worker.process", bind=True)
def process(
    self,
    job_id: str,
    message: str | None = None,
    filename: str | None = None,
    mime: str | None = None,
    content: bytes | None = None,
) -> dict:
    """
    Process a text job.

    Parameters
    ----------
    job_id : str
        Unique identifier for the job.
    message : str, optional
        Chat message (query flow).
    filename : str, optional
        Original filename (ingest flow).
    mime : str, optional
        MIME type of the uploaded file.
    content : bytes, optional
        Raw file bytes (ingest flow).
    """
    try:
        self.update_state(state="PROCESSING", meta={"job_id": job_id})

        # ── Ingest flow (file upload) ────────────────
        if content is not None:
            if mime == "application/pdf":
                text = extract_text_from_pdf(content)
            else:
                text = content.decode("utf-8", errors="replace")
            
            doc_id = filename or job_id
            result = ingest(doc_id, text)
            return {
                "job_id": job_id,
                "type": "ingest",
                "doc_id": doc_id,
                "chunks_stored": result["chunks"],
            }

        # ── Query flow (chat message) ────────────────
        if message:
            result = query(message)
            return {
                "job_id": job_id,
                "type": "query",
                "answer": result["answer"],
                "sources": result["sources"],
            }

        return {
            "job_id": job_id,
            "type": "error",
            "detail": "No message or content provided.",
        }

    except Exception as exc:
        logger.exception("Text worker failed for job %s", job_id)
        self.update_state(state="FAILURE", meta={"job_id": job_id, "error": str(exc)})
        raise
