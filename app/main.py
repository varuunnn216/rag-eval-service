from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import get_settings
from app.db import get_db_connection, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize PostgreSQL + pgvector schema on startup."""
    init_db()
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


@app.get("/health", tags=["system"])
def health_check() -> Dict[str, Any]:
    """Verify API and PostgreSQL + pgvector connectivity and corpus counts."""
    settings = get_settings()
    with get_db_connection() as conn:
        doc_count = conn.execute("SELECT COUNT(*) AS cnt FROM documents").fetchone()["cnt"]
        chunk_count = conn.execute("SELECT COUNT(*) AS cnt FROM chunks").fetchone()["cnt"]
        ext_row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return {
        "status": "ok",
        "pgvector_version": ext_row["extversion"] if ext_row else None,
        "documents_count": doc_count,
        "chunks_count": chunk_count,
        "embedding_model": settings.embedding_model,
        "generation_model": settings.generation_model,
        "reranker_model": settings.reranker_model,
    }
