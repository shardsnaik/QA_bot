"""
voice_ws/main.py

Real-time Voice AI — WebSocket orchestrator (FastAPI + Render).

Pipeline (concurrent, NOT sequential):
    microphone → STT (Groq Whisper) → LLM streaming (Groq) → TTS → speaker

All three stages overlap via asyncio — audio is streaming in while
tokens are streaming out and audio is streaming back.

Run:
    uvicorn voice_ws.main:app --host 0.0.0.0 --port 8000

Deploy:
    Render Web Service — set start command to above.

Env vars:
    GROQ_API_KEY        your Groq Cloud API key
    TTS_PROVIDER        "groq" | "openai" | "edge" (default: edge — free)
    OPENAI_API_KEY      only needed if TTS_PROVIDER=openai
    VAD_SILENCE_MS      silence threshold in ms (default: 800)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import wave
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("voice_ws")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv

import os
# Load explicitly from the backedn/.env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GROQ_STT_URL     = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_LLM_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TTS_URL     = "https://api.groq.com/openai/v1/audio/speech"

GROQ_STT_MODEL   = "whisper-large-v3-turbo"
GROQ_LLM_MODEL   = os.environ.get("GROQ_LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_TTS_MODEL   = "playai-tts"          # Groq TTS model
GROQ_TTS_VOICE   = "Aaliyah-PlayAI"      # one of Groq's supported voices

TTS_PROVIDER     = os.environ.get("TTS_PROVIDER", "edge")   # edge = free, no key needed
VAD_SILENCE_MS   = int(os.environ.get("VAD_SILENCE_MS", "800"))

SENTENCE_ENDINGS = {".", "!", "?", "؟", "。", "！", "？"}

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. "
    "Keep responses short and conversational — 1 to 3 sentences maximum. "
    "Avoid lists, markdown, or code blocks. Speak naturally."
)

# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# Standalone FastAPI app  (deployed independently on Render)
# ──────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,https://ragchatbot.sharadsnaik.in"
).split(",")

app = FastAPI(
    title="Voice AI — Real-Time WebSocket",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Internal router (keeps endpoint definitions clean)
router = APIRouter()


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "model": GROQ_LLM_MODEL, "service": "voice-ws"}


@app.get("/", tags=["Health"])
async def root():
    return {"service": "Voice AI WebSocket", "ws": "/ws/voice", "docs": "/docs"}


# ──────────────────────────────────────────────────────────────
# STT — Groq Whisper
# ──────────────────────────────────────────────────────────────

async def transcribe_audio(client: httpx.AsyncClient, pcm_bytes: bytes,
                            sample_rate: int = 16000) -> str:
    """Send PCM audio to Groq Whisper, return transcript string."""
    # Wrap raw PCM in a WAV container so Groq accepts it
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_buf.seek(0)

    response = await client.post(
        GROQ_STT_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", wav_buf, "audio/wav")},
        data={"model": GROQ_STT_MODEL, "response_format": "text"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text.strip()


# ──────────────────────────────────────────────────────────────
# LLM — Groq streaming
# ──────────────────────────────────────────────────────────────

async def stream_llm_tokens(
    client: httpx.AsyncClient,
    messages: list[dict],
) -> AsyncIterator[str]:
    """Yield LLM tokens one at a time via SSE streaming."""
    # Sanitise: drop blank turns, merge consecutive same-role turns
    # Groq returns 400 when any message has empty/whitespace content
    clean: list[dict] = []
    for msg in messages:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if clean and clean[-1]["role"] == msg["role"] == "assistant":
            clean[-1]["content"] += " " + content
        else:
            clean.append({"role": msg["role"], "content": content})

    if not clean:
        logger.warning("stream_llm_tokens: no valid messages after sanitisation")
        return

    payload = {
        "model":       GROQ_LLM_MODEL,
        "messages":    clean,
        "stream":      True,
        "max_tokens":  200,
        "temperature": 0.7,
    }

    async with client.stream(
        "POST", GROQ_LLM_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        json=payload,
        timeout=60,
    ) as resp:
        # Read full error body before raising — httpx streaming loses it otherwise
        if resp.status_code >= 400:
            body = await resp.aread()
            raise RuntimeError(
                f"Groq LLM {resp.status_code}: {body.decode()[:400]}"
            )
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                token = chunk["choices"][0]["delta"].get("content", "")
                if token:
                    yield token
            except (json.JSONDecodeError, KeyError):
                continue


# ──────────────────────────────────────────────────────────────
# Sentence chunker
# ──────────────────────────────────────────────────────────────

class SentenceChunker:
    """
    Buffer tokens and emit complete sentences when a boundary is detected.
    Boundaries: . ! ?
    """
    def __init__(self):
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        """Add a token, return list of complete sentences (may be empty)."""
        self._buf += token
        sentences = []
        while True:
            for i, ch in enumerate(self._buf):
                if ch in SENTENCE_ENDINGS:
                    # Check it's not a decimal (e.g. "3.14")
                    if ch == "." and i > 0 and i < len(self._buf) - 1:
                        if self._buf[i-1].isdigit() and self._buf[i+1].isdigit():
                            continue
                    sentence = self._buf[:i+1].strip()
                    self._buf = self._buf[i+1:].lstrip()
                    if sentence:
                        sentences.append(sentence)
                    break
            else:
                break
        return sentences

    def flush(self) -> str | None:
        """Return any remaining buffered text as a final chunk."""
        text = self._buf.strip()
        self._buf = ""
        return text if text else None


# ──────────────────────────────────────────────────────────────
# TTS providers
# ──────────────────────────────────────────────────────────────

async def tts_groq(client: httpx.AsyncClient, text: str) -> bytes:
    """Groq TTS — returns mp3 bytes."""
    resp = await client.post(
        GROQ_TTS_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": GROQ_TTS_MODEL, "input": text, "voice": GROQ_TTS_VOICE},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


async def tts_edge(text: str) -> bytes:
    """
    Microsoft Edge TTS — completely free, no API key.
    Uses edge-tts Python package.
    pip install edge-tts
    """
    try:
        import edge_tts
        import tempfile, os as _os

        tts = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        await tts.save(tmp.name)
        with open(tmp.name, "rb") as f:
            data = f.read()
        _os.unlink(tmp.name)
        return data
    except ImportError:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")


async def synthesize(client: httpx.AsyncClient, text: str) -> bytes:
    """Route to the configured TTS provider."""
    if TTS_PROVIDER == "groq":
        return await tts_groq(client, text)
    elif TTS_PROVIDER == "edge":
        return await tts_edge(text)
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {TTS_PROVIDER}")


# ──────────────────────────────────────────────────────────────
# WebSocket message helpers
# ──────────────────────────────────────────────────────────────

async def ws_send(ws: WebSocket, event: str, **payload):
    """Send a typed JSON event to the frontend."""
    await ws.send_text(json.dumps({"event": event, **payload}))


# ──────────────────────────────────────────────────────────────
# Voice Activity Detection (simple energy-based)
# ──────────────────────────────────────────────────────────────

class SimpleVAD:
    """
    Lightweight energy-based VAD.
    Detects silence by comparing RMS energy to a dynamic threshold.
    """
    SILENCE_THRESHOLD = 300    # RMS below this = silence (16-bit PCM)
    MIN_SPEECH_FRAMES = 3      # minimum chunks before we consider it real speech

    def __init__(self, silence_ms: int = VAD_SILENCE_MS, chunk_ms: int = 300):
        self._silence_chunks = silence_ms // chunk_ms
        self._silent_count   = 0
        self._speech_count   = 0
        self._in_speech      = False

    def process(self, pcm_chunk: bytes) -> tuple[bool, bool]:
        """
        Returns (is_speech, end_of_utterance).
        end_of_utterance = True when silence follows speech.
        """
        import struct
        samples = struct.unpack(f"{len(pcm_chunk)//2}h", pcm_chunk)
        rms     = (sum(s*s for s in samples) / len(samples)) ** 0.5

        is_speech = rms > self.SILENCE_THRESHOLD

        if is_speech:
            self._speech_count += 1
            self._silent_count  = 0
            if self._speech_count >= self.MIN_SPEECH_FRAMES:
                self._in_speech = True
        else:
            if self._in_speech:
                self._silent_count += 1

        end_of_utterance = (
            self._in_speech and
            self._silent_count >= self._silence_chunks
        )
        if end_of_utterance:
            self._reset()

        return is_speech, end_of_utterance

    def _reset(self):
        self._in_speech  = False
        self._speech_count = 0
        self._silent_count = 0


# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────

class VoiceSession:
    def __init__(self):
        self.conversation: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.audio_buffer: bytearray  = bytearray()
        self.vad = SimpleVAD()
        self.tts_queue: asyncio.Queue = asyncio.Queue()
        self.is_processing = False


# ──────────────────────────────────────────────────────────────
# Core pipeline coroutines
# ──────────────────────────────────────────────────────────────

async def llm_and_tts_pipeline(
    ws: WebSocket,
    client: httpx.AsyncClient,
    session: VoiceSession,
    transcript: str,
):
    """
    Run LLM streaming + sentence detection + TTS concurrently.
    Tokens stream in → sentences detected → TTS fires per sentence.
    """
    session.conversation.append({"role": "user", "content": transcript})
    await ws_send(ws, "transcript", text=transcript)

    chunker   = SentenceChunker()
    full_reply = ""
    tts_tasks  = []

    async def _tts_and_send(sentence: str):
        """Synthesize one sentence and stream audio back."""
        try:
            audio_bytes = await synthesize(client, sentence)
            b64         = base64.b64encode(audio_bytes).decode()
            await ws_send(ws, "audio_chunk", data=b64, format="mp3")
        except Exception as e:
            logger.warning("TTS failed for '%s': %s", sentence[:40], e)

    # Stream tokens from LLM
    async for token in stream_llm_tokens(client, session.conversation):
        full_reply += token
        await ws_send(ws, "token", text=token)

        # Check for complete sentences
        sentences = chunker.feed(token)
        for sentence in sentences:
            logger.info("TTS sentence: %s", sentence[:60])
            task = asyncio.create_task(_tts_and_send(sentence))
            tts_tasks.append(task)

    # Flush any remaining text
    remainder = chunker.flush()
    if remainder:
        task = asyncio.create_task(_tts_and_send(remainder))
        tts_tasks.append(task)

    # Wait for all TTS tasks to complete
    if tts_tasks:
        await asyncio.gather(*tts_tasks)

    # Only append assistant turn if it actually has content
    # Empty strings cause Groq 400 on the next request
    if full_reply.strip():
        session.conversation.append({"role": "assistant", "content": full_reply.strip()})

    # Keep conversation history bounded (last 10 turns + system prompt)
    system = session.conversation[:1]
    turns  = session.conversation[1:]
    if len(turns) > 20:
        session.conversation = system + turns[-20:]

    await ws_send(ws, "turn_end")


async def process_utterance(
    ws: WebSocket,
    client: httpx.AsyncClient,
    session: VoiceSession,
    pcm_bytes: bytes,
):
    """STT → LLM+TTS pipeline for one complete utterance."""
    try:
        await ws_send(ws, "processing", stage="stt")
        transcript = await transcribe_audio(client, pcm_bytes)

        if not transcript or len(transcript.split()) < 2:
            logger.info("Transcript too short (%r) — skipping", transcript)
            await ws_send(ws, "processing", stage="idle")
            return

        logger.info("Transcript: %s", transcript)
        await ws_send(ws, "processing", stage="llm")
        await llm_and_tts_pipeline(ws, client, session, transcript)
        await ws_send(ws, "processing", stage="idle")

    except Exception as e:
        logger.exception("Pipeline error: %s", e)
        await ws_send(ws, "error", message=str(e))
    finally:
        session.is_processing = False


# ──────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────

@router.websocket("/ws/voice")
async def voice_endpoint(ws: WebSocket):
    """
    Main WebSocket endpoint.

    Expected client messages (JSON):
        {"event": "audio_chunk", "data": "<base64 PCM>", "sample_rate": 16000}
        {"event": "end_of_speech"}   ← optional manual trigger
        {"event": "reset"}           ← clear conversation history

    Server events:
        {"event": "transcript",  "text": "..."}
        {"event": "token",       "text": "..."}
        {"event": "audio_chunk", "data": "<base64 mp3>", "format": "mp3"}
        {"event": "processing",  "stage": "stt|llm|idle"}
        {"event": "turn_end"}
        {"event": "error",       "message": "..."}
    """
    await ws.accept()
    logger.info("WebSocket connected: %s", ws.client)

    session = VoiceSession()

    async with httpx.AsyncClient() as client:
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")

                # ── Audio chunk from microphone ──────────────────
                if event == "audio_chunk":
                    b64_data    = msg.get("data", "")
                    sample_rate = msg.get("sample_rate", 16000)
                    pcm_chunk   = base64.b64decode(b64_data)

                    # Append to session buffer
                    session.audio_buffer.extend(pcm_chunk)

                    # VAD check
                    is_speech, end_of_utterance = session.vad.process(pcm_chunk)

                    if end_of_utterance and not session.is_processing:
                        pcm_snapshot         = bytes(session.audio_buffer)
                        session.audio_buffer = bytearray()
                        session.is_processing = True
                        # Fire pipeline — don't await (let it run concurrently)
                        asyncio.create_task(
                            process_utterance(ws, client, session, pcm_snapshot)
                        )

                # ── Manual end-of-speech trigger ────────────────
                elif event == "end_of_speech":
                    if session.audio_buffer and not session.is_processing:
                        pcm_snapshot          = bytes(session.audio_buffer)
                        session.audio_buffer  = bytearray()
                        session.is_processing = True
                        asyncio.create_task(
                            process_utterance(ws, client, session, pcm_snapshot)
                        )

                # ── Reset conversation ───────────────────────────
                elif event == "reset":
                    session.conversation = [
                        {"role": "system", "content": SYSTEM_PROMPT}
                    ]
                    session.audio_buffer = bytearray()
                    session.is_processing = False
                    await ws_send(ws, "reset_ack")
                    logger.info("Session reset")

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: %s", ws.client)
        except Exception as e:
            logger.exception("WebSocket error: %s", e)
            try:
                await ws_send(ws, "error", message=str(e))
            except Exception:
                pass



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Register routes into app
app.include_router(router)