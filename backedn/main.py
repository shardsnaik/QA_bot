"""
FastAPI entry point for the QA Bot backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.routes import router

app = FastAPI(
    title="QA Bot — RAG Pipeline",
    description="Multimodal QA Bot with hybrid retrieval (BM25 + Vector Search)",
    version="1.0.0",
)

# ── CORS (adjust origins for production) ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routes ─────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "QA Bot API is running", "docs": "/docs"}
