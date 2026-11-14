import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body

from app.config import get_settings
from app.generation.generator import answer_question
from app.ingestion.service import (
    delete_document,
    get_document_chunks,
    ingest_documents,
    list_documents,
)
from app.retrieval.hybrid import retrieve_candidates
from app.schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from eval.runner import run_evaluation_suite

router = APIRouter(prefix="/api/v1", tags=["rag"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(payload: IngestRequest) -> IngestResponse:
    """Chunk, embed, and index a batch of documents into PostgreSQL + pgvector."""
    return ingest_documents(payload.documents)


@router.get("/documents")
def list_documents_endpoint() -> List[Dict[str, Any]]:
    """List all indexed documents and their chunk counts."""
    return list_documents()


@router.get("/documents/{document_id}/chunks")
def get_document_chunks_endpoint(document_id: str) -> List[Dict[str, Any]]:
    """Return all indexed chunks for a specific document."""
    return get_document_chunks(document_id)


@router.delete("/documents/{document_id}")
def delete_document_endpoint(document_id: str) -> Dict[str, Any]:
    """Delete a document and its indexed chunks."""
    return delete_document(document_id)


@router.post("/search", response_model=SearchResponse)
def search_endpoint(payload: SearchRequest) -> SearchResponse:
    """Execute dense, sparse, or hybrid (RRF) search with optional cross-encoder reranking."""
    results, latency_ms = retrieve_candidates(
        query=payload.query,
        mode=payload.mode,
        top_k=payload.top_k,
        candidate_pool=payload.candidate_pool,
        use_reranker=payload.use_reranker,
        document_ids=payload.document_ids,
        min_sparse_score=payload.min_sparse_score,
    )
    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        use_reranker=payload.use_reranker,
        results=results,
        latency_ms=latency_ms,
    )


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest) -> AskResponse:
    """Retrieve, rerank, and generate a grounded answer with verified inline citations."""
    return answer_question(payload)


@router.get("/eval/report")
def get_eval_report_endpoint() -> Dict[str, Any]:
    """Return the latest evaluation benchmark report."""
    settings = get_settings()
    report_path = settings.project_root / "eval" / "reports" / "latest_eval_report.json"
    if not report_path.exists():
        return run_evaluation_suite()
    return json.loads(report_path.read_text(encoding="utf-8"))


@router.post("/eval/run")
def trigger_eval_run_endpoint(
    payload: Optional[Dict[str, Any]] = Body(default=None),
) -> Dict[str, Any]:
    """Re-run the evaluation suite across all 4 ablation configs or a single selected config."""
    selected = (payload or {}).get("config")
    return run_evaluation_suite(selected_config=selected)
