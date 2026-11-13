from typing import Any, Dict, List
from fastapi import APIRouter

from app.ingestion.service import ingest_documents, list_documents
from app.schemas import IngestRequest, IngestResponse

router = APIRouter(prefix="/api/v1", tags=["rag"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(payload: IngestRequest) -> IngestResponse:
    """Chunk, embed, and index a batch of documents into PostgreSQL + pgvector."""
    return ingest_documents(payload.documents)


@router.get("/documents")
def list_documents_endpoint() -> List[Dict[str, Any]]:
    """List all indexed documents and their chunk counts."""
    return list_documents()
