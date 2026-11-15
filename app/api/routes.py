import hmac
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.generation.generator import answer_question, select_context_chunks
from app.ingestion.service import (
    delete_document,
    get_document_chunks,
    ingest_documents,
    list_documents,
)
from app.retrieval.hybrid import retrieve_candidates, retrieve_candidates_with_timing
from app.schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from app.security import sanitize_ask_payload_for_demo, sanitize_documents_for_demo
from eval.question_bank import (
    add_manual_question,
    bulk_update_status,
    delete_question,
    load_all_questions,
    update_question,
)
from eval.question_generator import (
    generate_questions_for_document,
    get_generation_job,
    list_generation_jobs,
    start_generation_job,
)
from eval.runner import run_evaluation_suite

router = APIRouter(prefix="/api/v1", tags=["rag"])


def _require_admin(x_admin_token: Optional[str], authorization: Optional[str]) -> None:
    settings = get_settings()
    if not settings.demo_mode:
        return
    token = (settings.admin_token or "").strip()
    if not token:
        raise HTTPException(status_code=403, detail="Read-only in DEMO_MODE (admin token required)")
    hdr_token = (x_admin_token or "").strip()
    auth_hdr = (authorization or "").strip()
    if auth_hdr.lower().startswith("bearer "):
        auth_hdr = auth_hdr[7:].strip()
    ok = (
        (bool(hdr_token) and hmac.compare_digest(hdr_token, token))
        or (bool(auth_hdr) and hmac.compare_digest(auth_hdr, token))
    )
    if not ok:
        raise HTTPException(status_code=403, detail="Read-only in DEMO_MODE (admin token required)")


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(
    payload: IngestRequest,
    x_admin_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> IngestResponse:
    """Chunk, embed, and index a batch of documents into PostgreSQL + pgvector."""
    _require_admin(x_admin_token, authorization)
    return ingest_documents(payload.documents)


@router.get("/documents")
def list_documents_endpoint() -> List[Dict[str, Any]]:
    """List all indexed documents and their chunk counts."""
    return sanitize_documents_for_demo(list_documents())


@router.get("/documents/{document_id}/chunks")
def get_document_chunks_endpoint(document_id: str) -> List[Dict[str, Any]]:
    """Return all indexed chunks for a specific document."""
    return get_document_chunks(document_id)


@router.delete("/documents/{document_id}")
def delete_document_endpoint(
    document_id: str,
    x_admin_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Delete a document and its indexed chunks."""
    _require_admin(x_admin_token, authorization)
    return delete_document(document_id)


@router.post("/documents/{document_id}/delete")
def delete_document_post_endpoint(
    document_id: str,
    x_admin_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Delete a document and its indexed chunks (POST compatibility)."""
    _require_admin(x_admin_token, authorization)
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


@router.post("/ask")
def ask_endpoint(payload: AskRequest) -> Dict[str, Any]:
    """Retrieve, rerank, and generate a grounded answer with verified inline citations."""
    return sanitize_ask_payload_for_demo(answer_question(payload).to_dict())


@router.post("/ask/stream")
def ask_stream_endpoint(payload: AskRequest) -> StreamingResponse:
    """Stream retrieval, reranking, and generation progress over Server-Sent Events."""

    def event_stream():
        yield f"data: {json.dumps({'stage': 'retrieving'})}\n\n"
        raw_chunks, retrieval_ms, rerank_ms = retrieve_candidates_with_timing(
            query=payload.question,
            mode=payload.mode,
            top_k=payload.top_k,
            candidate_pool=payload.candidate_pool,
            use_reranker=payload.use_reranker,
            document_ids=payload.document_ids,
            query_expansion=payload.query_expansion,
        )
        yield f"data: {json.dumps({'stage': 'reranking', 'retrieval_ms': retrieval_ms, 'rerank_ms': rerank_ms})}\n\n"
        _annotated, selected_chunks, abstain_pre_llm = select_context_chunks(
            raw_chunks,
            min_score_override=payload.rerank_min_score,
            relative_factor_override=payload.rerank_relative_factor,
            max_chunks_override=payload.rerank_max_chunks,
        )
        yield f"data: {json.dumps({'stage': 'generating', 'selected_count': len(selected_chunks), 'abstain_pre_llm': abstain_pre_llm})}\n\n"
        resp = answer_question(payload)
        if not resp.abstained and not resp.error and resp.answer:
            words = resp.answer.split(" ")
            accum: List[str] = []
            step = max(2, len(words) // 8)
            for i in range(0, len(words), step):
                accum.extend(words[i : i + step])
                yield f"data: {json.dumps({'stage': 'delta', 'text': ' '.join(accum)})}\n\n"
                time.sleep(0.015)
        yield f"data: {json.dumps({'stage': 'done', 'response': sanitize_ask_payload_for_demo(resp.to_dict())}, default=str)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/eval/questions")
def list_eval_questions_endpoint() -> Dict[str, Any]:
    """List all golden and synthetic evaluation questions."""
    qs = load_all_questions()
    settings = get_settings()
    return {
        "questions": qs,
        "demo_mode": settings.demo_mode,
        "counts": {
            "total": len(qs),
            "golden": sum(1 for q in qs if q.get("source") == "golden"),
            "synthetic": sum(1 for q in qs if q.get("source") == "synthetic"),
            "approved": sum(1 for q in qs if q.get("status") == "approved"),
            "draft": sum(1 for q in qs if q.get("status") == "draft"),
            "rejected": sum(1 for q in qs if q.get("status") == "rejected"),
            "needs_review": sum(1 for q in qs if q.get("needs_review")),
        },
    }


@router.post("/eval/questions", status_code=201)
def create_eval_question_endpoint(
    payload: Dict[str, Any] = Body(...),
    x_admin_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_admin(x_admin_token, authorization)
    return add_manual_question(payload)


@router.post("/eval/questions/bulk")
def bulk_update_eval_questions_endpoint(
    payload: Dict[str, Any] = Body(...),
    x_admin_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_admin(x_admin_token, authorization)
    qids = [str(x) for x in (payload.get("ids") or [])]
    new_status = str(payload.get("status") or "approved")
    count = bulk_update_status(qids, new_status)
    return {"updated": count, "status": new_status}


@router.post("/eval/questions/{qid}")
def update_eval_question_endpoint(
    qid: str,
    payload: Dict[str, Any] = Body(...),
    x_admin_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_admin(x_admin_token, authorization)
    updated = update_question(qid, payload)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Question {qid} not found")
    return updated


@router.delete("/eval/questions/{qid}")
def delete_eval_question_endpoint(
    qid: str,
    x_admin_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_admin(x_admin_token, authorization)
    if delete_question(qid):
        return {"deleted": True, "id": qid}
    raise HTTPException(status_code=404, detail="Question not found or cannot delete golden question")


@router.post("/eval/generate")
def generate_eval_questions_endpoint(
    payload: Dict[str, Any] = Body(...),
    x_admin_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_admin(x_admin_token, authorization)
    doc_id = str(payload.get("doc_id") or "")
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required")
    per_chunk = int(payload.get("per_chunk", 2))
    include_unans = bool(payload.get("include_unanswerable", True))
    if payload.get("sync"):
        return generate_questions_for_document(
            doc_id, per_chunk=per_chunk, include_unanswerable=include_unans
        )
    return start_generation_job(doc_id, per_chunk=per_chunk, include_unanswerable=include_unans)


@router.get("/eval/generate/status")
def generate_eval_status_endpoint(job_id: str = Query(default="")) -> Dict[str, Any]:
    if job_id:
        job = get_generation_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return job
    return {"jobs": list_generation_jobs()}


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
    x_admin_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Re-run the evaluation suite across all 4 ablation configs or a single selected config."""
    _require_admin(x_admin_token, authorization)
    selected = (payload or {}).get("config")
    question_set = str((payload or {}).get("question_set") or "golden_only")
    doc_id = (payload or {}).get("doc_id") or None
    return run_evaluation_suite(
        selected_config=selected,
        question_set=question_set,
        doc_id=doc_id,
    )
