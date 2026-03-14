from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from utils.mime_detector import detect_mime_type
from pipelines.audio_pipeline  import run as run_audio_pipeline
import uuid
import logging
from routes.common import enforce_size, AUDIO_TYPES

logger = logging.getLogger(__name__)
router = APIRouter()

class VoiceResponse(BaseModel):
    job_id:     str
    task:       str
    transcript: str | None = None
    result:     str | None = None
    model:      str
    filename:   str | None = None
    extras:     dict | None = None

@router.post(
    "/voice",
    response_model=VoiceResponse,
    summary="Voice / Audio-to-text via Whisper-large-v3-turbo",
    description="Upload any supported audio file with an optional instruction.",
)
async def voice_endpoint(
    file: UploadFile = File(
        ...,
        description="Audio file — WAV, MP3, OGG, FLAC (max 50 MB)"
    ),
    message: str = Form(
        default="",
        description=(
            "Optional instruction. Leave blank for plain speech-to-text."
        ),
    ),
) -> VoiceResponse:
    content = await file.read()
    mime    = detect_mime_type(content, file.content_type)

    if mime not in AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Expected an audio file. Got MIME type: '{mime}'. "
                f"Accepted: {sorted(AUDIO_TYPES)}"
            ),
        )
    enforce_size(content, "audio")

    job_id = str(uuid.uuid4())
    logger.info(
        "[%s] /voice | file=%s | mime=%s | prompt='%s'",
        job_id, file.filename, mime, (message or "")[:60],
    )

    try:
        result = run_audio_pipeline(
            job_id=job_id,
            content=content,
            mime=mime,
            message=message or None,
        )
    except RuntimeError as exc:
        if "loading" in str(exc).lower():
            raise HTTPException(status_code=503, detail=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("[%s] Voice pipeline error", job_id)
        raise HTTPException(status_code=500, detail=f"Audio pipeline error: {exc}")

    return VoiceResponse(
        job_id=result.job_id,
        task=result.task,
        transcript=result.transcript,
        result=result.result,
        model=result.model,
        filename=file.filename,
        extras=result.extras,
    )
