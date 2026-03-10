"""
pipelines/audio_pipeline.py

Voice / Audio → Text pipeline using openai/whisper-large-v3-turbo
via HuggingFace Inference API (router.huggingface.co — correct 2025 URL).

Logic:
  • No message  →  Whisper transcription only, result = transcript text
  • message     →  Whisper transcription first, then Groq answers the
                   user's message using the transcript as context

Install:
    pip install huggingface_hub pydub requests
    apt-get install -y ffmpeg
"""

from __future__ import annotations

import io
import logging
import os
import requests
from dataclasses import dataclass
from pydantic import BaseModel
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

HF_TOKEN         = os.environ["HF_TOKEN"]
HF_ASR_MODEL     = "openai/whisper-large-v3-turbo"
# ✅ Correct 2025 URL — old api-inference.huggingface.co returns 410 Gone
HF_ASR_URL       = f"https://router.huggingface.co/hf-inference/models/{HF_ASR_MODEL}"

GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL       = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL         = "https://api.groq.com/openai/v1/chat/completions"

TARGET_SAMPLE_HZ = 16_000

# ──────────────────────────────────────────────────────────────
# Singleton HF InferenceClient
# ──────────────────────────────────────────────────────────────

_hf_client: InferenceClient | None = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
        logger.info("HuggingFace InferenceClient initialised → %s", HF_ASR_MODEL)
    return _hf_client


# ──────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────

class AudioPipelineResponse(BaseModel):
    job_id:     str
    transcript: str
    result:     str
    task:       str = "transcription"
    model:      str = HF_ASR_MODEL
    extras:     dict = {}

# ──────────────────────────────────────────────────────────────
# Audio pre-processing
# ──────────────────────────────────────────────────────────────

def _to_wav_bytes(audio_bytes: bytes, mime: str) -> tuple[bytes, str]:
    """
    Convert any audio format → 16 kHz mono WAV via pydub + ffmpeg.
    Returns (bytes, current_mime).
    """
    try:
        from pydub import AudioSegment

        fmt = (
            "mp3"  if "mpeg" in mime or "mp3" in mime else
            "wav"  if "wav"  in mime else
            "ogg"  if "ogg"  in mime else
            "flac" if "flac" in mime else
            "mp4"
        )
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        seg = seg.set_channels(1).set_frame_rate(TARGET_SAMPLE_HZ)

        buf = io.BytesIO()
        seg.export(buf, format="wav")
        wav = buf.getvalue()
        logger.info("Audio converted → 16 kHz mono WAV (%.1f KB)", len(wav) / 1024)
        return wav, "audio/wav"

    except ImportError:
        logger.warning("pydub not installed — sending original bytes as %s", mime)
        return audio_bytes, mime
    except Exception as exc:
        logger.warning("Audio conversion failed (%s) — sending original bytes as %s", exc, mime)
        return audio_bytes, mime


# ──────────────────────────────────────────────────────────────
# Whisper transcription
# ──────────────────────────────────────────────────────────────

def _call_whisper_client(audio_bytes: bytes, mime: str) -> str:
    """Primary: InferenceClient ASR → transcript string."""
    result = _get_hf_client().automatic_speech_recognition(
        audio=audio_bytes,
        model=HF_ASR_MODEL,
    )
    # Handle both object and dict returns
    if isinstance(result, dict):
        return (result.get("text", "") or "").strip()
    return (getattr(result, "text", "") or "").strip()


def _call_whisper_http(audio_bytes: bytes, mime: str) -> str:
    """Fallback: direct HTTP POST to router.huggingface.co."""
    response = requests.post(
        HF_ASR_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": mime},
        data=audio_bytes,
        timeout=120,
    )
    if response.status_code == 503:
        raise RuntimeError(
            "Whisper model is loading on HuggingFace (cold start). Retry in ~20 seconds."
        )
    if not response.ok:
        raise RuntimeError(f"HF ASR error {response.status_code}: {response.text[:300]}")
    return response.json().get("text", "").strip()


def _transcribe(audio_bytes: bytes, mime: str) -> str:
    """Transcribe audio → text. InferenceClient first, HTTP fallback."""
    try:
        return _call_whisper_client(audio_bytes, mime)
    except Exception as exc:
        logger.warning("InferenceClient failed (%s) — falling back to HTTP", exc)
        return _call_whisper_http(audio_bytes, mime)


# ──────────────────────────────────────────────────────────────
# Groq — only called when user provides a message
# ──────────────────────────────────────────────────────────────

def _groq_chat(transcript: str, user_message: str) -> str:
    """
    Answer the user's message using the transcript as context.
    The full transcript is injected into the system prompt so Groq
    can answer any question about the audio content.
    """
    if not GROQ_API_KEY:
        return "[GROQ_API_KEY not set — cannot process message]"

    system_prompt = (
        "You are a helpful audio assistant. "
        "The user has uploaded an audio file. Below is the full transcript of that audio.\n\n"
        f"### Audio Transcript\n{transcript}\n\n"
        "Answer the user's question or complete their instruction using only the transcript above. "
        "Be concise and accurate. Do not invent information not present in the transcript."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 800,
    }
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        err_msg = "[Groq Error]"
        if 'r' in locals() and r.text:
            try:
                detail = r.json().get("error", {}).get("message", r.text)
                err_msg = f"{err_msg} {detail}"
            except:
                err_msg = f"{err_msg} {r.text[:200]}"
        else:
            err_msg = f"{err_msg} {str(exc)}"
        logger.error(err_msg)
        return f"[{err_msg}]"


# ──────────────────────────────────────────────────────────────
# Public entry point  (called by audio_worker.py)
# ──────────────────────────────────────────────────────────────

def run(
    job_id:  str,
    content: bytes,
    mime:    str,
    message: str | None = None,
) -> AudioPipelineResponse:
    """
    Audio pipeline entry point.

    No message  →  Whisper transcription only, result = raw transcript
    message     →  Whisper transcription, then Groq answers the message
                   using the transcript as context
    """
    logger.info("[%s] Audio pipeline | mime=%s | size=%.1f KB | has_message=%s",
                job_id, mime, len(content) / 1024, bool(message))

    # Step 1 — convert to WAV
    wav_bytes, current_mime = _to_wav_bytes(content, mime)

    # Step 2 — transcribe with Whisper
    transcript = _transcribe(wav_bytes, current_mime)
    logger.info("[%s] Whisper transcript: %d chars", job_id, len(transcript))

    if not transcript:
        return AudioPipelineResponse(
            job_id=job_id,
            transcript="",
            result="[No speech detected in the audio]",
        )

    # Step 3 — decide output path
    if message and message.strip():
        # User sent a message → answer it using the transcript as context
        logger.info("[%s] Message provided → calling Groq", job_id)
        result = _groq_chat(transcript, message.strip())
    else:
        # No message → plain speech-to-text, return transcript directly
        logger.info("[%s] No message → returning raw transcript", job_id)
        result = transcript

    logger.info("[%s] Audio pipeline complete | result_len=%d", job_id, len(result))

    return AudioPipelineResponse(
        job_id=job_id,
        transcript=transcript,
        result=result,
    )