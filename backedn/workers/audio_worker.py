
"""
workers/audio_worker.py

Celery worker for audio/voice-to-text jobs.
Receives raw audio bytes from the router, calls audio_pipeline.run(),
and stores the structured result in the Celery result backend (Redis).
"""

from __future__ import annotations

import logging
from celery import current_app as celery_app
from pipelines.audio_pipeline import run as audio_run

logger = logging.getLogger(__name__)


@celery_app.task(name="workers.audio_worker.process", bind=True)
def process(
    self,
    job_id:   str,
    content:  bytes,
    mime:     str,
    filename: str | None = None,
    message:  str | None = None,
) -> dict:
    """
    Celery task — audio/voice to text.

    Parameters
    ----------
    job_id   : unique job id (from router)
    content  : raw audio bytes
    mime     : MIME type e.g. 'audio/wav', 'audio/mpeg'
    filename : original filename (metadata only)
    message  : optional user instruction for task resolution
               e.g. "translate this", "summarise", "give me timestamps"
               default → speech-to-text transcription
    """
    self.update_state(state="PROCESSING", meta={"job_id": job_id})
    logger.info(
        "[%s] Audio worker received job | file=%s | mime=%s | message='%s'",
        job_id, filename, mime, (message or "")[:80],
    )

    try:
        result = audio_run(
            job_id=job_id,
            content=content,
            mime=mime,
            message=message,
        )

        return {
            "job_id":     result.job_id,
            "type":       "audio_to_text",
            "task":       result.task,
            "transcript": result.transcript,
            "result":     result.result,
            "model":      result.model,
            "filename":   filename,
            "extras":     result.extras,
        }

    except Exception as exc:
        logger.exception("[%s] Audio worker failed", job_id)
        self.update_state(
            state="FAILURE",
            meta={"job_id": job_id, "error": str(exc)},
        )
        raise