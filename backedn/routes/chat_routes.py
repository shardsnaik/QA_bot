from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from utils.mime_detector import detect_mime_type
from queues.celery_config import celery_app
from pipelines.text_pipeline import query as run_rag_query, ingest as run_rag_ingest
from utils.pdf_extractor import extract_text_from_pdf
import uuid
import logging
from routes.common import JobResponse, enforce_size, resolve_modality

logger = logging.getLogger(__name__)
router = APIRouter()

class TextRequest(BaseModel):
    message: str

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
    modality = resolve_modality(mime)
    enforce_size(content, modality)

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
        
        # Step 2 — Direct Fallback for Text/PDF
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
        
        raise HTTPException(
            status_code=500, 
            detail=f"Celery/Redis error: {e}. Portfolios only support text/pdf fallback."
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
