from typing import Any, Dict, List
from fastapi import APIRouter

from app.ingestion.service import ingest_documents, list_documents
from app.retrieval.hybrid import retrieve_candidates
from app.schemas import IngestRequest, IngestResponse, SearchRequest, SearchResponse

router = APIRouter(prefix="/api/v1", tags=["rag"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(payload: IngestRequest) -> IngestResponse:
    """Chunk, embed, and index a batch of documents into PostgreSQL + pgvector."""
    return ingest_documents(payload.documents)


@router.get("/documents")
def list_documents_endpoint() -> List[Dict[str, Any]]:
    """List all indexed documents and their chunk counts."""
    return list_documents()


@router.post("/search", response_model=SearchResponse)
def search_endpoint(payload: SearchRequest) -> SearchResponse:
    """Execute dense, sparse, or hybrid (RRF) search over indexed document chunks."""
    candidates, latency_ms = retrieve_candidates(
        query=payload.query,
        mode=payload.mode,
        top_k=payload.top_k,
        candidate_pool=payload.candidate_pool,
    )
    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        use_reranker=False,
        results=candidates[: payload.top_k],
        latency_ms=latency_ms,
    )
