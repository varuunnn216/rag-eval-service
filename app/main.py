from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import get_settings
from app.db import get_db_connection, init_db
from app.ingestion.service import list_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize PostgreSQL + pgvector schema on startup when database is available."""
    try:
        init_db()
    except Exception as exc:
        logger.info("Running with in-memory pgvector/tsvector mirror (%s)", exc)
    yield


app = FastAPI(
    title="RAG Service with Evaluation Harness",
    description=(
        "FastAPI + pgvector RAG service featuring hybrid search (dense + BM25/tsvector RRF), "
        "local cross-encoder reranking, verifiable inline citations, and a 30-50 question eval harness."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_web_ui() -> FileResponse:
    """Serve the interactive RAG & Evaluation Harness Web UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


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
        "reranker_model": settings.reranker_model,
    }
