from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import get_settings
from app.db import get_db_connection, init_db
from app.ingestion.seed import load_corpus_documents
from app.ingestion.service import ingest_documents, list_documents
from app.security import (
    MAX_REQUEST_BODY_BYTES,
    SECURITY_HEADERS,
    check_rate_limit,
    friendly_error_payload,
    get_allowed_origins,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize PostgreSQL + pgvector schema and seed corpus on startup."""
    try:
        init_db()
    except Exception as exc:
        logger.info("Running with in-memory pgvector/tsvector mirror (%s)", exc)
    try:
        ingest_documents(load_corpus_documents())
    except Exception as exc:
        logger.warning("Startup corpus seeding skipped (%s)", exc)
    yield


_settings = get_settings()

app = FastAPI(
    title="RAG Service with Evaluation Harness",
    description=(
        "FastAPI + pgvector RAG service featuring hybrid search (dense + BM25/tsvector RRF), "
        "local cross-encoder reranking, verifiable inline citations, and a 72-question eval harness."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if _settings.demo_mode else "/docs",
    redoc_url=None if _settings.demo_mode else "/redoc",
    openapi_url=None if _settings.demo_mode else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Token", "Authorization"],
)


@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429, headers=SECURITY_HEADERS)
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_REQUEST_BODY_BYTES:
        return JSONResponse({"detail": "Request body too large"}, status_code=413, headers=SECURITY_HEADERS)
    try:
        response = await call_next(request)
    except Exception:
        err = friendly_error_payload()
        logger.exception("Unhandled error request_id=%s path=%s", err["request_id"], request.url.path)
        return JSONResponse(err, status_code=500, headers=SECURITY_HEADERS)
    for k, v in SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    return response

app.include_router(api_router)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_web_ui() -> FileResponse:
    """Serve the interactive RAG & Evaluation Harness Web UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/favicon.svg", include_in_schema=False)
@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "favicon.svg"), media_type="image/svg+xml")


@app.get("/health", tags=["system"])
def health_check() -> Dict[str, Any]:
    """Verify API and PostgreSQL + pgvector connectivity and corpus counts."""
    settings = get_settings()
    try:
        with get_db_connection() as conn:
            doc_count = conn.execute("SELECT COUNT(*) AS cnt FROM documents").fetchone()["cnt"]
            chunk_count = conn.execute("SELECT COUNT(*) AS cnt FROM chunks").fetchone()["cnt"]
            ext_row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            pgv_ver = ext_row["extversion"] if ext_row else None
            backend = "postgresql+pgvector"
    except Exception:
        docs = list_documents()
        doc_count = len(docs)
        chunk_count = sum(d.get("chunk_count", 0) for d in docs)
        pgv_ver = "in-memory-mirror"
        backend = "in-memory-hybrid"

    return {
        "status": "ok",
        "backend": backend,
        "pgvector_version": pgv_ver,
        "documents_count": doc_count,
        "chunks_count": chunk_count,
        "embedding_model": settings.embedding_model,
        "generation_model": settings.generation_model,
        "generator_model": settings.generator_model,
        "reranker_model": settings.reranker_model,
        "demo_mode": settings.demo_mode,
    }
