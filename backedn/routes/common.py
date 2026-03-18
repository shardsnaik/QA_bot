from fastapi import HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# MIME type mappings
# ──────────────────────────────────────────────────────────────

TEXT_TYPES  = {"text/plain", "text/markdown", "text/csv", "application/pdf"}
IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/flac", "audio/webm"}
VIDEO_TYPES = {"video/mp4", "video/mpeg", "video/webm", "video/quicktime"}

# ──────────────────────────────────────────────────────────────
# File size limits (bytes)
# ──────────────────────────────────────────────────────────────

SIZE_LIMITS = {
    "image": 10  * 1024 * 1024,   # 10 MB
    "audio": 50  * 1024 * 1024,   # 50 MB
    "video": 200 * 1024 * 1024,   # 200 MB
    "text":  5   * 1024 * 1024,   #  5 MB
}

# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    status:   str = "success"
    filename: str | None = None
    data:     dict | None = None

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def enforce_size(content: bytes, modality: str) -> None:
    limit = SIZE_LIMITS.get(modality, 0)
    if limit and len(content) > limit:
        mb = limit // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"{modality.capitalize()} files must be under {mb} MB."
        )

def resolve_modality(content_type: str) -> str:
    if content_type in TEXT_TYPES:
        return "text"
    if content_type in IMAGE_TYPES:
        return "image"
    if content_type in AUDIO_TYPES:
        return "audio"
    if content_type in VIDEO_TYPES:
        return "video"
    raise HTTPException(
        status_code=415,
        detail=f"Unsupported MIME type: '{content_type}'. "
               f"Accepted: text, image, audio, video."
    )
