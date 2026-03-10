"""
pipelines/image_pipeline.py

Vision pipeline — image + optional text prompt → text answer.

Model  : moonshotai/Kimi-K2.5   (HuggingFace free Inference API)
Flow   : bytes → base64 → HF /v1/chat/completions (OpenAI-compat) → answer

Install:
    pip install openai pillow          # openai SDK talks to HF endpoint
"""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from dataclasses import dataclass

from openai import AsyncOpenAI          # HF inference uses OpenAI-compat API
from PIL import Image
from groq import AsyncGroq
# from utils.config import HUGGING_FACE_API_KEY
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

HF_TOKEN        = os.environ["HF_TOKEN"]           # HuggingFace token (read scope)
HF_MODEL        = "moonshotai/Kimi-K2.5"
HF_BASE_URL     = "https://router.huggingface.co/v1"

MAX_TOKENS      = int(os.environ.get("IMAGE_MAX_TOKENS", "1024"))
MAX_IMAGE_PX    = int(os.environ.get("IMAGE_MAX_PX",     "1024"))  # resize before sending
DEFAULT_PROMPT  = "Describe this image in detail."

# ──────────────────────────────────────────────────────────────
# Singleton clientyy
# ──────────────────────────────────────────────────────────────

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=HF_TOKEN,
            base_url=HF_BASE_URL,
        )
        logger.info("HuggingFace AsyncOpenAI client initialised → %s", HF_MODEL)
    return _client

_client_backup: AsyncGroq | None = None

def _get_client_backup() -> AsyncGroq:
    global _client_backup
    if _client_backup is None:
        _client_backup = AsyncGroq()
    return _client_backup


# ──────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────

@dataclass
class ImagePipelineResponse:
    job_id:  str
    answer:  str
    model:   str = HF_MODEL
    prompt:  str = ""


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _resize_if_needed(image_bytes: bytes, mime: str) -> bytes:
    """
    Downscale the image so its longest side ≤ MAX_IMAGE_PX.
    Keeps aspect ratio. Returns original bytes if already small enough.
    """
    img = Image.open(BytesIO(image_bytes))
    w, h = img.size

    if max(w, h) <= MAX_IMAGE_PX:
        return image_bytes

    scale  = MAX_IMAGE_PX / max(w, h)
    new_wh = (int(w * scale), int(h * scale))
    img    = img.resize(new_wh, Image.LANCZOS)

    fmt = "JPEG" if "jpeg" in mime or "jpg" in mime else "PNG"
    buf = BytesIO()
    img.save(buf, format=fmt)
    resized = buf.getvalue()
    logger.info("Resized image %dx%d → %dx%d (%.1f KB)",
                w, h, new_wh[0], new_wh[1], len(resized) / 1024)
    return resized


def _to_data_url(image_bytes: bytes, mime: str) -> str:
    """Encode bytes as a base64 data URL for the vision API payload."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ──────────────────────────────────────────────────────────────
# Core generation
# ──────────────────────────────────────────────────────────────

async def _call_vision_with_failover(data_url: str, prompt: str) -> tuple[str, str]:
    """
    Send image + prompt to Kimi-K2.5 via HuggingFace OpenAI-compat endpoint.
    If it fails, fall back to Groq Llama-4-Scout (text-only).
    Returns (answer, model_name).
    """
    client = _get_client()
    
    try:
        logger.info("Attempting primary vision model: %s", HF_MODEL)
        response = await client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip(), HF_MODEL

    except Exception as exc:
        logger.warning("Primary vision model (%s) failed: %s. Falling back to Groq Vision...", HF_MODEL, exc)
        backup_client = _get_client_backup()
        # Use Groq's vision model for true failover
        backup_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        try:
            response = await backup_client.chat.completions.create(
                model=backup_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.3, # Keep temperature consistent with primary
            )
            answer = response.choices[0].message.content.strip()
            return answer, backup_model
        except Exception as groq_exc:
            logger.error("Vision failover to Groq also failed: %s", groq_exc)
            raise RuntimeError(f"Vision model error: {exc}. Backup error: {groq_exc}")


# ──────────────────────────────────────────────────────────────
# Public entry point  (called by image_worker.py)
# ──────────────────────────────────────────────────────────────

async def run(
    job_id:  str,
    content: bytes,
    mime:    str,
    message: str | None = None,
) -> ImagePipelineResponse:
    """
    Full image-to-text pipeline:
        bytes → resize → base64 → Kimi-K2.5 → answer

    Parameters
    ----------
    job_id  : unique job identifier
    content : raw image bytes
    mime    : MIME type e.g. 'image/png', 'image/jpeg'
    message : optional user prompt; falls back to DEFAULT_PROMPT
    """
    prompt = (message or DEFAULT_PROMPT).strip()
    logger.info("[%s] Image pipeline started | mime=%s | prompt='%s'",
                job_id, mime, prompt[:80])

    # Step 1 — Resize to stay within model / bandwidth limits
    image_bytes = _resize_if_needed(content, mime)

    # Step 2 — Encode to base64 data URL
    data_url = _to_data_url(image_bytes, mime)

    # Step 3 — Call model with failover
    answer, model_name = await _call_vision_with_failover(data_url, prompt)
    logger.info("[%s] Image pipeline complete | model=%s | answer_len=%d", 
                job_id, model_name, len(answer))

    return ImagePipelineResponse(
        job_id=job_id,
        answer=answer,
        model=model_name,
        prompt=prompt,
    )