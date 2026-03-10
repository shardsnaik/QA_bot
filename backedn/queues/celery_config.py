"""
Celery application factory.
Uses Redis as broker and result backend.
"""

from celery import Celery
from utils.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

celery_app = Celery(
    "qa_bot",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "workers.text_worker",
        "workers.image_workers", 
        "workers.audio_worker"
    ],
)

# ── Task routing ────────────────────────────────
celery_app.conf.task_routes = {
    "workers.text_worker.process": {"queue": "text"},
}

# ── Serialisation ───────────────────────────────
celery_app.conf.accept_content = ["json", "pickle"]
celery_app.conf.task_serializer = "pickle"        # bytes support for file content
celery_app.conf.result_serializer = "json"
celery_app.conf.result_expires = 3600             # 1 hour
