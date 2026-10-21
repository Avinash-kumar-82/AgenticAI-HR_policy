import hmac
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from backend.core.config import get_settings
from backend.rag.workflow import ask
from backend.rag.vectorstore import add_documents
from backend.services.ingestion import (
    SUPPORTED,
    chunk_documents,
    load_file,
)
from backend.services.audit import write_audit


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["API"])
settings = get_settings()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
DEFAULT_ADMIN_KEY = "change-me-in-production"


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
    }


@router.post("/chat")
def chat(payload: ChatRequest):
    question = payload.question.strip()

    if len(question) < 2:
        raise HTTPException(
            status_code=422,
            detail="Question must contain at least 2 characters.",
        )

    try:
        result = ask(question)

        write_audit(
            question,
            result.get("source_used", "unknown"),
            result.get("trace", []),
        )

        return {
            "answer": result["answer"],
            "source_used": result.get("source_used", "unknown"),
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", question),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unexpected error while processing chat request")

        error_name = type(exc).__name__.lower()
        error_message = str(exc).lower()

        if any(
            term in error_name or term in error_message
            for term in ("ratelimit", "rate limit", "too many requests")
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "The AI provider rate limit was reached. "
                    "Please try again later."
                ),
            ) from exc

        if any(
            term in error_name or term in error_message
            for term in (
                "timeout",
                "connection",
                "serviceunavailable",
                "apiconnection",
            )
        ):
            raise HTTPException(
                status_code=502,
                detail=(
                    "An AI or retrieval service request failed. "
                    "Please try again later."
                ),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process your question. "
                "Check server logs for details."
            ),
        ) from exc


def _index_file(dest: Path, original_name: str) -> tuple[int, int]:
    """Blocking work: parse, chunk, embed, upsert. Runs in a thread."""
    docs = load_file(dest, source_name=original_name)
    chunks = chunk_documents(docs)

    if not chunks:
        raise ValueError("No text could be extracted from the uploaded file.")

    ids = add_documents(chunks)
    return len(chunks), len(ids)


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default="", alias="X-Admin-Key"),
):
    configured_key = settings.admin_api_key

    if not configured_key or configured_key == DEFAULT_ADMIN_KEY:
        logger.error("ADMIN_API_KEY has not been securely configured")
        raise HTTPException(
            status_code=503,
            detail="Document ingestion is not configured securely.",
        )

    # compare_digest raises TypeError on non-ASCII str; compare as bytes.
    if not hmac.compare_digest(
        x_admin_key.encode("utf-8"),
        configured_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key.",
        )

    filename = Path(file.filename or "").name

    if not filename or filename in {".", ".."}:
        raise HTTPException(
            status_code=400,
            detail="A valid filename is required.",
        )

    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Supported file types: {', '.join(sorted(SUPPORTED))}",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{uuid4().hex}{suffix}"
    dest = upload_dir / stored_name

    saved_file = False
    indexed = False

    try:
        # Read at most the configured limit plus one byte.
        content = await file.read(MAX_UPLOAD_SIZE + 1)

        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum upload size is 10 MB.",
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        dest.write_bytes(content)
        saved_file = True

        try:
            chunk_count, id_count = await run_in_threadpool(
                _index_file, dest, filename
            )
        except ValueError as exc:
            # Parsing problems (empty text, scanned PDF, unsupported content)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        indexed = True

        logger.info(
            "Document indexed: original_file=%s stored_file=%s chunks=%s",
            filename,
            stored_name,
            chunk_count,
        )

        return {
            "message": "Document indexed successfully",
            "file": filename,
            "chunks": chunk_count,
            "ids_created": id_count,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Document ingestion failed: %s", filename)
        raise HTTPException(
            status_code=500,
            detail=(
                "Document ingestion failed. "
                "Check server logs for details."
            ),
        ) from exc

    finally:
        await file.close()

        # Keep indexed files for traceability. Remove failed local copies.
        if saved_file and not indexed:
            dest.unlink(missing_ok=True)