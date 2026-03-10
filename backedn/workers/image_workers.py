"""
workers/image_worker.py

Celery worker for image-to-text jobs.
Receives raw image bytes from the router, calls image_pipeline.run(),
and stores the result in the Celery result backend (Redis).
"""

from __future__ import annotations

import asyncio
import logging

from celery import current_app as celery_app
from pipelines.image_pipelin import run as image_run

logger = logging.getLogger(__name__)


@celery_app.task(name="workers.image_worker.process", bind=True)
def process(
    self,
    job_id:   str,
    content:  bytes,
    mime:     str,
    filename: str | None = None,
    message:  str | None = None,
) -> dict:
    """
    Celery task — image-to-text.

    Parameters
    ----------
    job_id   : unique job id (from router)
    content  : raw image bytes
    mime     : MIME type string e.g. 'image/png'
    filename : original upload filename (metadata only)
    message  : optional user prompt for the vision model
    """
    self.update_state(state="PROCESSING", meta={"job_id": job_id})
    logger.info("[%s] Image worker received job | file=%s | mime=%s",
                job_id, filename, mime)

    try:
        # image_pipeline.run() is async; workers run in a sync Celery context
        result = asyncio.get_event_loop().run_until_complete(
            image_run(
                job_id=job_id,
                content=content,
                mime=mime,
                message=message,
            )
        )

        return {
            "job_id":   result.job_id,
            "type":     "image_to_text",
            "answer":   result.answer,
            "model":    result.model,
            "prompt":   result.prompt,
            "filename": filename,
        }

    except Exception as exc:
        logger.exception("[%s] Image worker failed", job_id)
        self.update_state(
            state="FAILURE",
            meta={"job_id": job_id, "error": str(exc)},
        )
        raise