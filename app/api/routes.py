import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
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
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from app.security import (
    ADMIN_COOKIE_NAME,
    ADMIN_SESSION_TTL_SECONDS,
    check_login_rate_limit,
    create_admin_session_value,
    is_admin_authenticated,
    sanitize_ask_payload_for_demo,
    sanitize_documents_for_demo,
    verify_admin_token,
)
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


def _require_admin(request: Request) -> None:
    if not is_admin_authenticated(request.headers, request.cookies):
        raise HTTPException(
            status_code=401,
            detail="Read-only in DEMO_MODE (admin sign-in or X-Admin-Token required)",
        )


def _is_https_request(request: Request) -> bool:
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "").lower()
    return proto == "https" or os.getenv("VERCEL") == "1"


@router.get("/admin/status")
def admin_status_endpoint(request: Request) -> Dict[str, Any]:
    settings = get_settings()
    return {
        "demo_mode": settings.demo_mode,
        "is_admin": is_admin_authenticated(request.headers, request.cookies),
    }


@router.post("/admin/login")
def admin_login_endpoint(
    request: Request,
    response: Response,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    if not check_login_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Please wait 15 minutes and try again.",
        )
    token = str(payload.get("token") or "")
    if not verify_admin_token(token):
        raise HTTPException(status_code=401, detail="Invalid admin token")
    session_val = create_admin_session_value()
    response.set_cookie(
        key=ADMIN_COOKIE_NAME,
        value=session_val,
        max_age=ADMIN_SESSION_TTL_SECONDS,
        path="/",
        httponly=True,
        samesite="strict",
        secure=_is_https_request(request),
    )
    return {"authenticated": True, "is_admin": True}


@router.post("/admin/logout")
def admin_logout_endpoint(request: Request, response: Response) -> Dict[str, Any]:
    response.delete_cookie(
        key=ADMIN_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="strict",
        secure=_is_https_request(request),
    )
    return {"authenticated": False, "is_admin": False}


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(payload: IngestRequest, request: Request) -> IngestResponse:
    """Chunk, embed, and index a batch of documents into PostgreSQL + pgvector."""
    _require_admin(request)
    return ingest_documents(payload.documents)


@router.get("/documents")
def list_documents_endpoint(request: Request) -> List[Dict[str, Any]]:
    """List all indexed documents and their chunk counts."""
    is_admin = is_admin_authenticated(request.headers, request.cookies)
    return sanitize_documents_for_demo(list_documents(), is_admin=is_admin)


@router.get("/documents/{document_id}/chunks")
def get_document_chunks_endpoint(document_id: str) -> List[Dict[str, Any]]:
    """Return all indexed chunks for a specific document."""
    return get_document_chunks(document_id)


@router.delete("/documents/{document_id}")
def delete_document_endpoint(document_id: str, request: Request) -> Dict[str, Any]:
    """Delete a document and its indexed chunks."""
    _require_admin(request)
    return delete_document(document_id)


@router.post("/documents/{document_id}/delete")
def delete_document_post_endpoint(document_id: str, request: Request) -> Dict[str, Any]:
    """Delete a document and its indexed chunks (POST compatibility)."""
    _require_admin(request)
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
def ask_endpoint(payload: AskRequest, request: Request) -> Dict[str, Any]:
    """Retrieve, rerank, and generate a grounded answer with verified inline citations."""
    is_admin = is_admin_authenticated(request.headers, request.cookies)
    return sanitize_ask_payload_for_demo(answer_question(payload).to_dict(), is_admin=is_admin)


@router.post("/ask/stream")
def ask_stream_endpoint(payload: AskRequest, request: Request) -> StreamingResponse:
    """Stream retrieval, reranking, and generation progress over Server-Sent Events."""
    is_admin = is_admin_authenticated(request.headers, request.cookies)

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
        yield f"data: {json.dumps({'stage': 'done', 'response': sanitize_ask_payload_for_demo(resp.to_dict(), is_admin=is_admin)}, default=str)}\n\n"

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
def create_eval_question_endpoint(request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    _require_admin(request)
    return add_manual_question(payload)


@router.post("/eval/questions/bulk")
def bulk_update_eval_questions_endpoint(request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    _require_admin(request)
    qids = [str(x) for x in (payload.get("ids") or [])]
    new_status = str(payload.get("status") or "approved")
    count = bulk_update_status(qids, new_status)
    return {"updated": count, "status": new_status}


@router.post("/eval/questions/{qid}")
def update_eval_question_endpoint(
    qid: str,
    request: Request,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    _require_admin(request)
    updated = update_question(qid, payload)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Question {qid} not found")
    return updated


@router.delete("/eval/questions/{qid}")
def delete_eval_question_endpoint(qid: str, request: Request) -> Dict[str, Any]:
    _require_admin(request)
    if delete_question(qid):
        return {"deleted": True, "id": qid}
    raise HTTPException(status_code=404, detail="Question not found or cannot delete golden question")


@router.post("/eval/generate")
def generate_eval_questions_endpoint(request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    _require_admin(request)
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
    request: Request,
    payload: Optional[Dict[str, Any]] = Body(default=None),
) -> Dict[str, Any]:
    """Re-run the evaluation suite across all 4 ablation configs or a single selected config."""
    _require_admin(request)
    selected = (payload or {}).get("config")
    question_set = str((payload or {}).get("question_set") or "golden_only")
    doc_id = (payload or {}).get("doc_id") or None
    return run_evaluation_suite(
        selected_config=selected,
        question_set=question_set,
        doc_id=doc_id,
    )
