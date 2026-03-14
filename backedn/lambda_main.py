"""
FastAPI entry point for the QA Bot backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.routes import router
from mangum import Mangum
app = FastAPI(
    title="QA Bot — RAG + Multimodal API",
    description="Multimodal QA Bot with hybrid retrieval (BM25 + Vector Search)",
    version="1.0.0",
)
# ── CORS (adjust origins for production) ─────────
# ALLOWED_ORIGINS env var lets you add origins without code changes.
# On Render set: ALLOWED_ORIGINS=https://ragchatbot.sharadsnaik.in,https://your-voice-render-url.onrender.com
_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
_extra = [o.strip() for o in _origins_env.split(",") if o.strip()]
ALLOWED_ORIGINS = list(dict.fromkeys([
    "http://localhost:3000",
    "http://localhost:5173",
    "https://ragchatbot.sharadsnaik.in",
] + _extra))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routes ─────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "QA Bot API is running", "docs": "/docs"}

handler = Mangum(app)