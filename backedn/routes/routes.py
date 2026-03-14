from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from utils.mime_detector import detect_mime_type
from queues.celery_config import celery_app
from pipelines.text_pipeline import query as run_rag_query, ingest as run_rag_ingest
from pipelines.image_pipeline import run as run_image_pipeline
from pipelines.audio_pipeline  import run as run_audio_pipeline

from utils.pdf_extractor import extract_text_from_pdf
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ──────────────────────────────────────────────
# MIME type mappings
# ──────────────────────────────────────────────

TEXT_TYPES  = {"text/plain", "text/markdown", "text/csv", "application/pdf"}
IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/flac"}
VIDEO_TYPES = {"video/mp4", "video/mpeg", "video/webm", "video/quicktime"}

# ──────────────────────────────────────────────
# File size limits (bytes)
# ──────────────────────────────────────────────

SIZE_LIMITS = {
    "image": 10  * 1024 * 1024,   # 10 MB
    "audio": 50  * 1024 * 1024,   # 50 MB
    "video": 200 * 1024 * 1024,   # 200 MB
    "text":  5   * 1024 * 1024,   #  5 MB
}


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class TextRequest(BaseModel):
    message: str


class JobResponse(BaseModel):
    status:   str = "success"
    filename: str | None = None
    data:     dict | None = None


class VisionResponse(BaseModel):
    job_id:   str
    answer:   str
    model:    str
    prompt:   str
    filename: str

class VoiceResponse(BaseModel):
    job_id:     str
    task:       str
    transcript: str | None = None
    result:     str | None = None
    model:      str
    filename:   str | None = None
    extras:     dict | None = None

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _enforce_size(content: bytes, modality: str) -> None:
    limit = SIZE_LIMITS.get(modality, 0)
    if limit and len(content) > limit:
        mb = limit // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"{modality.capitalize()} files must be under {mb} MB."
        )


def _resolve_modality(content_type: str) -> str:
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


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@router.post("/upload-direct", summary="Process file upload via Celery (synchronous response)")
async def route_upload_direct(
    file: UploadFile = File(...),
    message: str | None = None
) -> JobResponse:
    """
    Accepts a file upload, dispatches to Celery, and waits for the result.
    If `message` is provided, the worker handles ingestion and returns an optional query result.
    """
    content = await file.read()
    mime = detect_mime_type(content, file.content_type)
    modality = _resolve_modality(mime)
    _enforce_size(content, modality)

    job_id = str(uuid.uuid4())
    task_name = f"workers.{modality}_worker.process"
    
    # Step 1 — Dispatch to Celery if available
    try:
        task = celery_app.send_task(
            task_name,
            kwargs={
                "job_id":   job_id,
                "filename": file.filename,
                "mime":     mime,
                "content":  content,
                "message":  message,
            },
            queue=modality,
        )
        result = task.get(timeout=30)  # Wait for result
        return JobResponse(status="success", filename=file.filename, data=result)

    except Exception as e:
        logger.warning("Celery/Redis failed or timed out: %s. Attempting direct fallback...", e)
        
        # Step 2 — Direct Fallback for Text/PDF (the most common upload-direct case)
        if modality == "text":
            try:
                if mime == "application/pdf":
                    text = extract_text_from_pdf(content)
                else:
                    text = content.decode("utf-8", errors="replace")
                
                doc_id = file.filename or job_id
                ingest_result = run_rag_ingest(doc_id, text)
                return JobResponse(
                    status="success",
                    filename=file.filename,
                    data={
                        "job_id": job_id,
                        "type": "ingest",
                        "doc_id": doc_id,
                        "chunks_stored": ingest_result["chunks"],
                        "note": "Processed via direct fallback (Celery/Redis unavailable)"
                    }
                )
            except Exception as direct_exc:
                logger.exception("Direct fallback also failed")
                raise HTTPException(status_code=500, detail=f"Upload failed: {e}. Fallback error: {direct_exc}")
        
        # For other modalities, we don't have a simple direct sync fallback here yet
        # (Image/Audio have their own dedicated endpoints /vision and /voice for sync)
        raise HTTPException(
            status_code=500, 
            detail=f"Celery/Redis error: {e}. Please ensure Redis is running or use /vision / /voice for other file types."
        )


@router.post("/chat-direct", summary="Direct chat (bypasses Celery/Redis)")
async def route_chat_direct(data: TextRequest) -> dict:
    """
    Processes the RAG query directly using the pipeline (FastAPI sync).
    """
    try:
        result = run_rag_query(data.message)
        return {
            "status": "success",
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vision",
    response_model=VisionResponse,
    summary="Image-to-text via Kimi-K2.5 (moonshotai)",
    description=(
        "Upload any supported image (PNG, JPEG, WEBP, GIF) with an optional text prompt. "
        "Returns a natural-language answer generated by moonshotai/Kimi-K2.5 "
        "via the HuggingFace free Inference API.\n\n"
        "**Direct** — runs the pipeline in-process (no Celery hop). "
        "Suitable for images ≤ 10 MB."
    ),
)
async def vision_endpoint(
    file:    UploadFile = File(..., description="Image file (PNG / JPEG / WEBP / GIF)"),
    message: str        = Form(
        default="Describe this image in detail.",
        description="Question or instruction for the vision model.",
    ),
) -> VisionResponse:
    """
    Image-text-to-text endpoint.

    Flow:
        image upload + prompt
              │
              ▼
        MIME validation + size check
              │
              ▼
        image_pipeline.run()  ← resize → base64 → Kimi-K2.5
              │
              ▼
        VisionResponse (answer, model, sources)
    """
    # ── Read & validate ──────────────────────────────────────
    content = await file.read()

    mime = detect_mime_type(content, file.content_type)

    if mime not in IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Expected an image file. Got MIME type: '{mime}'. "
                   f"Accepted: {sorted(IMAGE_TYPES)}",
        )

    _enforce_size(content, "image")

    # ── Run vision pipeline directly (no queue needed for sync response) ──
    job_id = str(uuid.uuid4())
    logger.info("[%s] /vision called | file=%s | mime=%s | prompt='%s'",
                job_id, file.filename, mime, message[:80])

    try:
        result = await run_image_pipeline(
            job_id=job_id,
            content=content,
            mime=mime,
            message=message,
        )
    except Exception as exc:
        logger.exception("[%s] Vision pipeline error", job_id)
        raise HTTPException(status_code=500, detail=f"Vision model error: {exc}")

    return VisionResponse(
        job_id=result.job_id,
        answer=result.answer,
        model=result.model,
        prompt=result.prompt,
        filename=file.filename,
    )

@router.post(
    "/voice",
    response_model=VoiceResponse,
    summary="Voice / Audio-to-text via Whisper-large-v3-turbo",
    description="""
Upload any supported audio file with an optional instruction.

**Supported tasks** (auto-detected from your message):

| Instruction example                        | Task performed              |
|--------------------------------------------|-----------------------------|
| *(nothing / default)*                      | Speech-to-text transcription|
| "translate this to English"                | Transcribe + translate      |
| "summarise the audio"                      | Transcribe + LLM summary    |
| "what language is this?"                   | Language detection          |
| "give me timestamps" / "make subtitles"    | SRT subtitle generation     |

Model: **openai/whisper-large-v3-turbo** (HuggingFace free Inference API)
""",
)
async def voice_endpoint(
    file: UploadFile = File(
        ...,
        description="Audio file — WAV, MP3, OGG, FLAC (max 50 MB)"
    ),
    message: str = Form(
        default="",
        description=(
            "Optional instruction. Leave blank for plain speech-to-text. "
            "Examples: 'translate to English', 'summarise', "
            "'give me timestamps', 'what language is spoken?'"
        ),
    ),
) -> VoiceResponse:
    """
    Audio-to-text endpoint.

    Flow:
        audio bytes + optional instruction
              │
              ▼
        MIME validation + size check
              │
              ▼
        resolve_task(message)           ← keyword-based task detection
              │
              ▼
        audio_pipeline.run()
          ├─ _to_wav_bytes()            ← convert to 16 kHz mono WAV
          ├─ _call_whisper()            ← HF Whisper-large-v3-turbo
          └─ post-process by task
                ├─ TRANSCRIBE           → raw transcript
                ├─ TRANSLATE            → English translation (Groq cleanup)
                ├─ SUMMARISE            → bullet-point summary via Groq
                ├─ DETECT_LANGUAGE      → language name + confidence
                └─ TIMESTAMPS           → SRT subtitle file content
              │
              ▼
        VoiceResponse
    """
    # ── Validate MIME ────────────────────────────────────────
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
    _enforce_size(content, "audio")

    # ── Resolve & log task ───────────────────────────────────
    job_id       = str(uuid.uuid4())
    # resolved     = resolve_task(message or None)
    logger.info(
        "[%s] /voice | file=%s | mime=%s | task=%s | prompt='%s'",
        job_id, file.filename, mime, (message or "")[:60],
    )

    # ── Run pipeline ─────────────────────────────────────────
    try:
        result = run_audio_pipeline(
            job_id=job_id,
            content=content,
            mime=mime,
            message=message or None,
        )
    except RuntimeError as exc:
        # Surface HF cold-start 503 as a retryable 503
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
