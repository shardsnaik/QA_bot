# Multimodal AI + Real-Time Voice AI

## Checkout ✅✅ => https://ragchatbot.sharadsnaik.in/

A production-grade multimodal chatbot with **RAG (Retrieval-Augmented Generation)**, **vision**, **audio transcription**, and **real-time voice-to-voice conversation** — all built on free-tier APIs.

---

## Stack at a Glance

| Layer | Technology |
|---|---|
| API Framework | FastAPI (async) |
| Task Queue | Celery + Redis |
| Vector Store | Pinecone (cloud serverless) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| LLM | Groq — `llama3-8b-8192` (free tier) |
| STT (batch) | HuggingFace — `openai/whisper-large-v3-turbo` |
| STT (real-time) | Groq — `whisper-large-v3-turbo` |
| Vision | HuggingFace — `moonshotai/Kimi-K2.5` |
| TTS | `edge-tts` (free, no key) or Groq TTS |
| Frontend | React + CSS Modules |
| Backend Deploy | Render |
| Frontend Deploy | Netlify |

---

## Project Structure

```
project/
│
├── router/
│   ├── main.py       # FastAPI app entry point
│   └── routes.py     # All HTTP endpoints
models
│   └── voice_to_voice.py  #realtime voice-to-voice
│
├── pipelines/
│   ├── text_pipeline.py         # RAG: embed → Pinecone → Groq LLM
│   ├── image_pipeline.py        # Vision: image → Kimi-K2.5 → answer
│   └── audio_pipeline.py        # Audio: Whisper → optional Groq
│
├── workers/
│   ├── text_worker.py   # Celery worker: ingest + RAG query
│   ├── image_worker.py  # Celery worker: vision
│   └── audio_worker.py  # Celery worker: audio transcription
│
├── voice_ws/
│   └── main.py           # Real-time WebSocket voice pipeline
│
├── queues/
│   └── celery_config.py  # Celery + Redis config
│
├── utils/
│   ├── mime_detector.py      # MIME type detection (magic bytes)
│   ├── pdf_extractor.py      # PDF → text
│   ├── text_chunker.py       # Chunk documents for ingestion
│   ├── embedding_manager.py  # Sentence-transformers wrapper
│   ├── vector_store.py       # Pinecone upsert / query
│   ├── bm25_retriever.py     # Keyword search (hybrid RAG)
│   ├── llm_client.py         # Groq LLM client
│   └── config.py             # All env var loading
│
└── frontend/                    # React app
    ├── src/
    │   ├── hooks/
    │   │   └── useVoiceSocket.js
    │   ├── components/
    │   │   ├── AudioVisualiser.jsx / .module.css
    │   │   ├── MessageList.jsx   / .module.css
    │   │   ├── MicButton.jsx     / .module.css
    │   │   └── StatusBar.jsx     / .module.css
    │   ├── VoiceApp.jsx
    │   └── VoiceApp.module.css
    └── public/
        └── index.html
```

---

## Environment Variables

Create a `.env` file in the project root. **Never commit this file.**

```env
# ── Required ────────────────────────────────────────────────
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ── Pinecone ─────────────────────────────────────────────────
PINECONE_INDEX_NAME=rag-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# ── RAG / LLM ────────────────────────────────────────────────
GROQ_MODEL=llama3-8b-8192
EMBED_MODEL=all-MiniLM-L6-v2
RAG_TOP_K=5
RAG_MAX_TOKENS=1024

# ── Vision ───────────────────────────────────────────────────
IMAGE_MAX_TOKENS=1024
IMAGE_MAX_PX=1024

# ── Real-time Voice ──────────────────────────────────────────
GROQ_LLM_MODEL=llama3-8b-8192
TTS_PROVIDER=edge               # "edge" = free  |  "groq" = Groq TTS
VAD_SILENCE_MS=800

# ── Celery / Redis ───────────────────────────────────────────
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# ── CORS (comma-separated) ───────────────────────────────────
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

### Where to get each key

| Key | URL |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → API Keys |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token (read scope) |

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Redis (local or cloud)
- ffmpeg (for audio conversion)

```bash
# Ubuntu / WSL
sudo apt-get install -y ffmpeg redis-server

# macOS
brew install ffmpeg redis
```

### Backend

```bash
git clone <your-repo>
cd project

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install fastapi uvicorn httpx groq \
            pinecone-client sentence-transformers \
            huggingface_hub pydub edge-tts \
            celery redis python-multipart \
            openai pillow requests tiktoken
```

### Frontend

```bash
cd frontend
npm install
```

---

## Running Locally

Open **4 terminals**:

```bash
# Terminal 1 — Redis
redis-server

# Terminal 2 — FastAPI backend
uvicorn router.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3 — Celery workers (all queues)
celery -A queues.celery_config.celery_app worker \
       --queues=text,image,audio,video \
       --concurrency=4 --loglevel=info

# Terminal 4 — React frontend
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

---

## API Endpoints

### HTTP Endpoints (`/api/v1/`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/chat-direct` | RAG text query — JSON `{"message": "..."}` |
| `POST` | `/api/v1/upload-direct` | Upload file (text/PDF/image/audio) via Celery |
| `POST` | `/api/v1/vision` | Image + optional prompt → Kimi-K2.5 answer |
| `POST` | `/api/v1/voice` | Audio file + optional query → Whisper + Groq |
| `GET` | `/api/v1/status/{job_id}` | Poll async Celery job |
| `GET` | `/api/v1/health` | Health check |

### WebSocket

| Endpoint | Description |
|---|---|
| `ws://localhost:8000/ws/voice` | Real-time push-to-talk voice conversation |

#### WebSocket Message Protocol

**Client → Server:**
```json
{ "event": "audio_chunk",   "data": "<base64 PCM>", "sample_rate": 16000 }
{ "event": "end_of_speech" }
{ "event": "reset" }
```

**Server → Client:**
```json
{ "event": "transcript",  "text": "What is the capital of France?" }
{ "event": "token",       "text": "Paris" }
{ "event": "audio_chunk", "data": "<base64 mp3>", "format": "mp3" }
{ "event": "processing",  "stage": "stt|llm|idle" }
{ "event": "turn_end" }
{ "event": "error",       "message": "..." }
```

---

## Pipelines

### 1. Text RAG Pipeline

```
User query
    │
    ▼
Embed query (all-MiniLM-L6-v2, 384-dim)
    │
    ▼
Pinecone vector search (cosine, top-k=5)
    │
    ▼
Build prompt (transcript + retrieved chunks)
    │
    ▼
Groq llama3-8b-8192 → answer
```

**Ingest documents:**
```python
from pipelines.text_pipeline import ingest_documents

ingest_documents([
    {"id": "doc1-p0", "text": "...", "metadata": {"source": "manual.pdf"}},
])
```

### 2. Vision Pipeline

```
Image upload (PNG / JPEG / WEBP)
    │
    ▼
Resize to max 1024px (Pillow)
    │
    ▼
Base64 encode → data URL
    │
    ▼
moonshotai/Kimi-K2.5 (HuggingFace free inference)
    │
    ▼
Text answer
```

### 3. Audio Pipeline (batch)

```
Audio upload (WAV / MP3 / OGG / FLAC)
    │
    ▼
Convert to 16kHz mono WAV (pydub + ffmpeg)
    │
    ▼
HuggingFace whisper-large-v3-turbo
(router.huggingface.co — correct 2025 URL)
    │
    ├─ No message  →  raw transcript
    └─ message     →  Groq LLM answers using transcript as context
```

### 4. Real-Time Voice Pipeline

```
Microphone (push-to-talk)
    │  PCM chunks over WebSocket every ~256ms
    ▼
SimpleVAD (energy-based silence detection)
    │  utterance complete → fire asyncio.create_task
    ▼
Groq Whisper STT (~0.5–1.5s)
    │
    ▼
Groq LLM streaming (~0.5s first token)
    │  tokens stream in concurrently with TTS
    ▼
SentenceChunker (detects . ! ? boundaries)
    │  per sentence → asyncio.create_task
    ▼
edge-tts / Groq TTS
    │
    ▼
base64 MP3 → WebSocket → AudioContext queue → speaker
```

> All three stages (STT → LLM → TTS) overlap. The user hears the first sentence
> while the LLM is still generating the second.

**Expected latency:**

| Stage | Time |
|---|---|
| STT | 0.5–1.5 s |
| LLM first token | ~0.5 s |
| TTS per sentence | ~0.8–1.5 s |
| **User hears first response** | **~2–3 s** |

---

## Ingesting Your Documents

```bash
# Create a small ingest script
python - << 'EOF'
from pipelines.text_pipeline import ingest_documents
import os

docs = []
for filename in os.listdir("./docs"):
    if filename.endswith(".txt"):
        with open(f"./docs/{filename}") as f:
            text = f.read()
        docs.append({
            "id": filename,
            "text": text,
            "metadata": {"source": filename}
        })

ingest_documents(docs)
print(f"Indexed {len(docs)} documents.")
EOF
```

For PDFs, use `utils/pdf_extractor.py` to extract text first, then chunk and ingest.

---

## Deployment

### Backend — Render

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set **Start Command:**
   ```bash
   uvicorn router.main:app --host 0.0.0.0 --port $PORT
   ```
4. Add all environment variables from `.env` in the Render dashboard
5. Add a **Redis** instance on Render and set `CELERY_BROKER_URL`

For Celery workers, create a separate Render **Background Worker**:
```bash
celery -A queues.celery_config.celery_app worker --queues=text,image,audio --concurrency=2 --loglevel=info
```

### Frontend — Netlify

1. Build the React app:
   ```bash
   cd frontend && npm run build
   ```
2. Drag the `dist/` folder to [netlify.com/drop](https://netlify.com/drop)
3. Or connect your GitHub repo and set:
   - **Build command:** `npm run build`
   - **Publish directory:** `dist`
4. Set environment variable:
   ```
   VITE_WS_URL=wss://your-render-app.onrender.com/ws/voice
   ```

> In `useVoiceSocket.js`, replace the `WS_URL` constant with:
> ```js
> const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/voice";
> ```

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `410 Gone` on HF ASR | Old `api-inference.huggingface.co` URL | Use `router.huggingface.co/hf-inference/models/...` |
| Groq `400 Bad Request` on LLM | Empty message content in history | Sanitise messages before sending (already fixed in `stream_llm_tokens`) |
| `503` on HF Whisper | Model cold-starting on free tier | Retry after ~20 seconds |
| Single-word transcripts firing LLM | VAD cutting off too early | Increase `VAD_SILENCE_MS` to `1000`–`1200` |
| Audio not playing in browser | `AudioContext` suspended (autoplay policy) | Already handled — context resumes on first user gesture |
| CORS error from React | Origins not whitelisted | Add your Netlify URL to `ALLOWED_ORIGINS` |

---

## Free Tier Limits

| Service | Free Limit |
|---|---|
| Groq API | 14,400 req/day, 30 req/min |
| Pinecone | 1 index, 2GB storage, 100K vectors |
| HuggingFace Inference | Rate-limited, cold starts on inactivity |
| edge-tts | Unlimited (Microsoft Edge TTS, no key) |
| Render | 750 hrs/month free, spins down after 15 min idle |

---

## License

MIT