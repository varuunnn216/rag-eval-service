from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

from app.config import get_settings
from app.generation.generator import (
    ABSTENTION_MESSAGE,
    LLMGenerationError,
    SYSTEM_PROMPT,
    answer_question,
    call_gemini_llm,
    estimate_token_count,
    format_context_prompt,
    select_context_chunks,
    verify_answer_citations,
)
from app.ingestion.seed import load_corpus_documents
from app.ingestion.service import (
    cleanup_expired_sandbox_documents,
    delete_document,
    get_document_chunks,
    get_session_sandbox_usage,
    ingest_documents,
    list_documents,
)
from app.retrieval.hybrid import retrieve_candidates, retrieve_candidates_with_timing
from app.schemas import AskRequest, AskResponse, DocumentInput
from app.security import (
    ADMIN_COOKIE_NAME,
    ADMIN_SESSION_TTL_SECONDS,
    MAX_REQUEST_BODY_BYTES,
    SANDBOX_MAX_CHUNKS_PER_SESSION,
    SECURITY_HEADERS,
    VISITOR_COOKIE_NAME,
    build_session_cookie_header,
    check_ask_rate_limit,
    check_daily_ingest_cap,
    check_daily_llm_cap,
    check_ingest_rate_limit,
    check_login_rate_limit,
    check_rate_limit,
    create_admin_session_value,
    friendly_error_payload,
    get_or_create_visitor_session,
    is_admin_authenticated,
    parse_cookies,
    sanitize_ask_payload_for_demo,
    sanitize_documents_for_demo,
    strip_control_chars,
    validate_sandbox_upload,
    verify_admin_token,
    verify_visitor_session_value,
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

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger(__name__)


def _is_admin_authorized(headers: Any) -> bool:
    return is_admin_authenticated(headers)


def _get_health_payload(headers: Any = None) -> Dict[str, Any]:
    settings = get_settings()
    visitor_sid = None
    if headers is not None:
        cookies = parse_cookies(headers.get("Cookie") or headers.get("cookie"))
        visitor_sid = verify_visitor_session_value(cookies.get(VISITOR_COOKIE_NAME))
    docs = list_documents(session_id=visitor_sid)
    doc_count = len(docs)
    chunk_count = sum(d.get("chunk_count", 0) for d in docs)
    return {
        "status": "ok",
        "backend": "pgvector-hybrid",
        "pgvector_version": "0.8.0",
        "documents_count": doc_count,
        "chunks_count": chunk_count,
        "embedding_model": settings.embedding_model,
        "generation_model": settings.generation_model,
        "generator_model": settings.generator_model,
        "reranker_model": settings.reranker_model,
        "demo_mode": settings.demo_mode,
        "is_admin": is_admin_authenticated(headers) if headers is not None else not settings.demo_mode,
    }


def _parse_ask_request(body: Dict[str, Any]) -> AskRequest:
    doc_ids = body.get("document_ids")
    if isinstance(doc_ids, list):
        doc_ids = [str(x) for x in doc_ids if x]
    else:
        doc_ids = None
    return AskRequest(
        question=str(body.get("question", "")),
        mode=body.get("mode", "hybrid"),
        top_k=int(body.get("top_k", 5)),
        candidate_pool=int(body.get("candidate_pool", 20)),
        use_reranker=bool(body.get("use_reranker", True)),
        query_expansion=bool(body.get("query_expansion", True)),
        document_ids=doc_ids,
        scope_mode=str(body.get("scope_mode") or "all"),
        rerank_min_score=(
            float(body["rerank_min_score"])
            if body.get("rerank_min_score") is not None
            else None
        ),
        rerank_relative_factor=(
            float(body["rerank_relative_factor"])
            if body.get("rerank_relative_factor") is not None
            else None
        ),
        rerank_max_chunks=(
            int(body["rerank_max_chunks"])
            if body.get("rerank_max_chunks") is not None
            else None
        ),
        llm_model=(
            str(body["llm_model"]).strip()
            if body.get("llm_model")
            else None
        ),
        temperature=(
            float(body["temperature"])
            if body.get("temperature") is not None
            else None
        ),
    )


class RAGWebRequestHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP handler serving the Web UI and /api/v1/* endpoints."""

    server_version = ""
    sys_version = ""

    def _send_security_headers(self) -> None:
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)

    def _send_json(self, payload: Any, status: int = 200, extra_headers: Dict[str, str] | None = None) -> None:
        raw = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._send_security_headers()
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, file_path: Path, extra_headers: Dict[str, str] | None = None) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"detail": "Not found"}, status=404)
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        if file_path.suffix == ".svg":
            mime = "image/svg+xml"
        elif file_path.suffix == ".woff2":
            mime = "font/woff2"
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        if file_path.parent.name == "fonts":
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self._send_security_headers()
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(content)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > MAX_REQUEST_BODY_BYTES:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_sse_event(self, payload: Dict[str, Any]) -> None:
        line = f"data: {json.dumps(payload, default=str)}\n\n".encode("utf-8")
        self.wfile.write(line)
        self.wfile.flush()

    def _visitor_session(self) -> tuple[str, Dict[str, str]]:
        is_https = (self.headers.get("X-Forwarded-Proto") or "").lower() == "https" or os.getenv("VERCEL") == "1"
        sid, set_cookie = get_or_create_visitor_session(self.headers, is_https=is_https)
        extra = {"Set-Cookie": set_cookie} if set_cookie else {}
        return sid, extra

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            _, extra = self._visitor_session()
            self._send_file(STATIC_DIR / "index.html", extra_headers=extra)
            return
        if path == "/favicon.ico" or path == "/favicon.svg":
            self._send_file(STATIC_DIR / "favicon.svg")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            static_root = STATIC_DIR.resolve()
            target = (static_root / rel).resolve()
            if target.is_relative_to(static_root):
                self._send_file(target)
                return
            self._send_json({"detail": "Forbidden"}, status=403)
            return
        if path == "/health":
            sid, extra = self._visitor_session()
            self._send_json(_get_health_payload(self.headers), extra_headers=extra)
            return
        if path == "/api/v1/cron/cleanup":
            cron_secret = (os.getenv("CRON_SECRET") or "").strip()
            auth_hdr = (self.headers.get("Authorization") or "").strip()
            if cron_secret:
                expected = f"Bearer {cron_secret}"
                if not hmac.compare_digest(auth_hdr, expected) and not _is_admin_authorized(self.headers):
                    self._send_json({"detail": "Unauthorized cron request"}, status=401)
                    return
            removed = cleanup_expired_sandbox_documents()
            self._send_json({"cleaned_up": removed})
            return
        if path == "/api/v1/admin/status":
            settings = get_settings()
            self._send_json({"demo_mode": settings.demo_mode, "is_admin": _is_admin_authorized(self.headers)})
            return
        if path == "/api/v1/documents":
            sid, extra = self._visitor_session()
            self._send_json(
                sanitize_documents_for_demo(list_documents(session_id=sid), is_admin=_is_admin_authorized(self.headers)),
                extra_headers=extra,
            )
            return
        if path.startswith("/api/v1/documents/") and path.endswith("/chunks"):
            parts = path.split("/")
            if len(parts) >= 5:
                sid, extra = self._visitor_session()
                doc_id = unquote(parts[4])
                self._send_json(get_document_chunks(doc_id, session_id=sid), extra_headers=extra)
                return
        if path == "/api/v1/eval/questions":
            qs = load_all_questions()
            settings = get_settings()
            self._send_json(
                {
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
            )
            return
        if path == "/api/v1/eval/generate/status":
            from urllib.parse import parse_qs

            q_params = parse_qs(parsed.query)
            job_id = (q_params.get("job_id") or [""])[0]
            if job_id:
                job = get_generation_job(job_id)
                if job is None:
                    self._send_json({"detail": "Unknown job_id"}, status=404)
                    return
                self._send_json(job)
            else:
                self._send_json({"jobs": list_generation_jobs()})
            return
        if path == "/api/v1/eval/report":
            settings = get_settings()
            report_path = settings.project_root / "eval" / "reports" / "latest_eval_report.json"
            if not report_path.exists():
                self._send_json(run_evaluation_suite())
            else:
                self._send_json(json.loads(report_path.read_text(encoding="utf-8")))
            return

        self._send_json({"detail": f"Unknown route: {path}"}, status=404)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/v1/eval/questions/"):
            if not _is_admin_authorized(self.headers):
                self._send_json({"detail": "Read-only in DEMO_MODE (admin token required)"}, status=401)
                return
            qid = unquote(path[len("/api/v1/eval/questions/") :].strip("/"))
            if qid and delete_question(qid):
                self._send_json({"deleted": True, "id": qid})
                return
            self._send_json({"detail": "Question not found or cannot delete golden question"}, status=404)
            return
        if path.startswith("/api/v1/documents/"):
            is_admin = _is_admin_authorized(self.headers)
            sid, extra = self._visitor_session()
            doc_id = unquote(path[len("/api/v1/documents/") :].strip("/"))
            if doc_id:
                try:
                    self._send_json(delete_document(doc_id, session_id=sid, is_admin=is_admin), extra_headers=extra)
                except PermissionError as exc:
                    self._send_json({"detail": str(exc)}, status=401, extra_headers=extra)
                except FileNotFoundError as exc:
                    self._send_json({"detail": str(exc)}, status=404, extra_headers=extra)
                return
        self._send_json({"detail": f"Unknown route: {path}"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        client_ip = self.client_address[0] if self.client_address else "unknown"
        if not check_rate_limit(client_ip):
            self._send_json({"detail": "Rate limit exceeded"}, status=429)
            return
        try:
            body = self._read_json_body()
            if path == "/api/v1/admin/login":
                if not check_login_rate_limit(client_ip):
                    self._send_json({"detail": "Too many login attempts. Please wait 15 minutes."}, status=429)
                    return
                token = str(body.get("token") or "")
                if not verify_admin_token(token):
                    self._send_json({"detail": "Invalid admin token"}, status=401)
                    return
                is_https = (self.headers.get("X-Forwarded-Proto") or "").lower() == "https"
                cookie_hdr = build_session_cookie_header(create_admin_session_value(), secure=is_https)
                self._send_json(
                    {"authenticated": True, "is_admin": True},
                    status=200,
                    extra_headers={"Set-Cookie": cookie_hdr},
                )
                return

            if path == "/api/v1/admin/logout":
                is_https = (self.headers.get("X-Forwarded-Proto") or "").lower() == "https"
                cookie_hdr = build_session_cookie_header("", max_age=0, secure=is_https)
                self._send_json(
                    {"authenticated": False, "is_admin": False},
                    status=200,
                    extra_headers={"Set-Cookie": cookie_hdr},
                )
                return

            if path == "/api/v1/ask":
                settings = get_settings()
                is_admin = _is_admin_authorized(self.headers)
                sid, extra = self._visitor_session()
                if settings.demo_mode and not is_admin:
                    if not check_ask_rate_limit(client_ip):
                        self._send_json({"detail": "Question rate limit reached (20 per hour). Please try again later."}, status=429, extra_headers=extra)
                        return
                    cap_ok, cap_msg = check_daily_llm_cap()
                    if not cap_ok:
                        self._send_json({"detail": cap_msg}, status=429, extra_headers=extra)
                        return
                req = _parse_ask_request(body)
                resp = answer_question(req, session_id=sid)
                self._send_json(sanitize_ask_payload_for_demo(resp.to_dict(), is_admin=is_admin), extra_headers=extra)
                return

            if path == "/api/v1/ask/stream":
                settings = get_settings()
                is_admin = _is_admin_authorized(self.headers)
                sid, extra = self._visitor_session()
                if settings.demo_mode and not is_admin:
                    if not check_ask_rate_limit(client_ip):
                        self._send_json({"detail": "Question rate limit reached (20 per hour). Please try again later."}, status=429, extra_headers=extra)
                        return
                    cap_ok, cap_msg = check_daily_llm_cap()
                    if not cap_ok:
                        self._send_json({"detail": cap_msg}, status=429, extra_headers=extra)
                        return
                req = _parse_ask_request(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                for k, v in extra.items():
                    self.send_header(k, v)
                self.end_headers()

                self._send_sse_event({"stage": "retrieving"})
                t_start = time.perf_counter()
                raw_chunks, retrieval_ms, rerank_ms = retrieve_candidates_with_timing(
                    query=req.question,
                    mode=req.mode,
                    top_k=req.top_k,
                    candidate_pool=req.candidate_pool,
                    use_reranker=req.use_reranker,
                    document_ids=req.document_ids,
                    query_expansion=req.query_expansion,
                    session_id=sid,
                    scope_mode=req.scope_mode,
                )
                self._send_sse_event(
                    {"stage": "reranking", "retrieval_ms": retrieval_ms, "rerank_ms": rerank_ms}
                )
                annotated_chunks, selected_chunks, abstain_pre_llm = select_context_chunks(
                    raw_chunks,
                    min_score_override=req.rerank_min_score,
                    relative_factor_override=req.rerank_relative_factor,
                    max_chunks_override=req.rerank_max_chunks,
                )
                self._send_sse_event(
                    {
                        "stage": "generating",
                        "selected_count": len(selected_chunks),
                        "abstain_pre_llm": abstain_pre_llm,
                    }
                )
                resp = answer_question(req, session_id=sid)
                # Stream answer tokens progressively for smooth visual feedback
                if not resp.abstained and not resp.error and resp.answer:
                    words = resp.answer.split(" ")
                    accum = []
                    step = max(2, len(words) // 8)
                    for i in range(0, len(words), step):
                        accum.extend(words[i : i + step])
                        self._send_sse_event({"stage": "delta", "text": " ".join(accum)})
                        time.sleep(0.018)
                self._send_sse_event({"stage": "done", "response": sanitize_ask_payload_for_demo(resp.to_dict(), is_admin=_is_admin_authorized(self.headers))})
                return

            if path == "/api/v1/search":
                query = str(body.get("query", ""))
                mode = body.get("mode", "hybrid")
                top_k = int(body.get("top_k", 5))
                candidate_pool = int(body.get("candidate_pool", 20))
                use_reranker = bool(body.get("use_reranker", True))
                doc_ids = body.get("document_ids")
                if isinstance(doc_ids, list):
                    doc_ids = [str(x) for x in doc_ids if x]
                else:
                    doc_ids = None
                min_sparse = (
                    float(body["min_sparse_score"])
                    if body.get("min_sparse_score") is not None
                    else None
                )
                sid, extra = self._visitor_session()
                scope_mode = str(body.get("scope_mode") or "all")
                results, latency_ms = retrieve_candidates(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    candidate_pool=candidate_pool,
                    use_reranker=use_reranker,
                    document_ids=doc_ids,
                    min_sparse_score=min_sparse,
                    session_id=sid,
                    scope_mode=scope_mode,
                )
                self._send_json(
                    {
                        "query": query,
                        "mode": mode,
                        "use_reranker": use_reranker,
                        "results": [r.to_dict() for r in results],
                        "latency_ms": latency_ms,
                    },
                    extra_headers=extra,
                )
                return

            if path == "/api/v1/ingest":
                is_admin = _is_admin_authorized(self.headers)
                raw_docs = body.get("documents")
                if raw_docs is None and "content" in body:
                    raw_docs = [body]
                docs = [
                    DocumentInput(
                        id=str(d.get("id") or d.get("document_id") or ""),
                        document_id=str(d.get("document_id") or d.get("id") or ""),
                        title=str(d.get("title") or ""),
                        source=str(d.get("source") or ""),
                        content=str(d["content"]),
                        metadata=dict(d.get("metadata") or {}),
                    )
                    for d in (raw_docs or [])
                ]
                if is_admin:
                    res = ingest_documents(docs)
                    self._send_json(res.to_dict())
                    return

                sid, extra = self._visitor_session()
                if not check_ingest_rate_limit(client_ip):
                    self._send_json({"detail": "Upload rate limit reached (5 per hour). Please try again later."}, status=429, extra_headers=extra)
                    return
                cap_ok, cap_msg = check_daily_ingest_cap()
                if not cap_ok:
                    self._send_json({"detail": cap_msg}, status=429, extra_headers=extra)
                    return
                if len(docs) != 1:
                    self._send_json({"detail": "Sandbox supports uploading 1 document at a time."}, status=400, extra_headers=extra)
                    return
                doc = docs[0]
                doc.content = strip_control_chars(doc.content)
                doc.title = strip_control_chars(doc.title)
                filename = str((doc.metadata or {}).get("filename") or doc.source or "")
                cur_docs, cur_chunks = get_session_sandbox_usage(sid)
                ok, err_msg = validate_sandbox_upload(filename, doc.content, cur_docs, cur_chunks)
                if not ok:
                    self._send_json({"detail": err_msg}, status=400, extra_headers=extra)
                    return
                prefix = f"sandbox_{sid[:12]}_"
                if not doc.id.startswith(prefix):
                    doc.id = f"{prefix}{doc.id}"
                    doc.document_id = doc.id
                expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
                try:
                    res = ingest_documents(
                        [doc],
                        session_id=sid,
                        expires_at=expires_at,
                        existing_session_chunks=cur_chunks,
                        max_session_chunks=SANDBOX_MAX_CHUNKS_PER_SESSION,
                    )
                    self._send_json(res.to_dict(), extra_headers=extra)
                except ValueError as exc:
                    self._send_json({"detail": str(exc)}, status=400, extra_headers=extra)
                return

            if path.startswith("/api/v1/documents/") and path.endswith("/delete"):
                is_admin = _is_admin_authorized(self.headers)
                sid, extra = self._visitor_session()
                parts = path.split("/")
                if len(parts) >= 5:
                    doc_id = unquote(parts[4])
                    try:
                        self._send_json(delete_document(doc_id, session_id=sid, is_admin=is_admin), extra_headers=extra)
                    except PermissionError as exc:
                        self._send_json({"detail": str(exc)}, status=401, extra_headers=extra)
                    except FileNotFoundError as exc:
                        self._send_json({"detail": str(exc)}, status=404, extra_headers=extra)
                    return

            if path == "/api/v1/eval/generate":
                if not _is_admin_authorized(self.headers):
                    self._send_json({"detail": "Generation disabled in DEMO_MODE (admin token required)"}, status=401)
                    return
                doc_id = str(body.get("doc_id") or "")
                if not doc_id:
                    self._send_json({"detail": "doc_id is required"}, status=400)
                    return
                per_chunk = int(body.get("per_chunk", 2))
                include_unans = bool(body.get("include_unanswerable", True))
                if body.get("sync"):
                    res = generate_questions_for_document(
                        doc_id, per_chunk=per_chunk, include_unanswerable=include_unans
                    )
                    self._send_json(res)
                else:
                    job = start_generation_job(
                        doc_id, per_chunk=per_chunk, include_unanswerable=include_unans
                    )
                    self._send_json(job, status=202)
                return

            if path == "/api/v1/eval/questions":
                if not _is_admin_authorized(self.headers):
                    self._send_json({"detail": "Read-only in DEMO_MODE (admin token required)"}, status=401)
                    return
                rec = add_manual_question(body)
                self._send_json(rec, status=201)
                return

            if path == "/api/v1/eval/questions/bulk":
                if not _is_admin_authorized(self.headers):
                    self._send_json({"detail": "Read-only in DEMO_MODE (admin token required)"}, status=401)
                    return
                qids = [str(x) for x in (body.get("ids") or [])]
                new_status = str(body.get("status") or "approved")
                count = bulk_update_status(qids, new_status)
                self._send_json({"updated": count, "status": new_status})
                return

            if path.startswith("/api/v1/eval/questions/"):
                if not _is_admin_authorized(self.headers):
                    self._send_json({"detail": "Read-only in DEMO_MODE (admin token required)"}, status=401)
                    return
                qid = unquote(path[len("/api/v1/eval/questions/") :].strip("/"))
                updated = update_question(qid, body)
                if updated is None:
                    self._send_json({"detail": f"Question {qid} not found"}, status=404)
                    return
                self._send_json(updated)
                return

            if path == "/api/v1/eval/run":
                if not _is_admin_authorized(self.headers):
                    self._send_json({"detail": "Eval run disabled in DEMO_MODE (admin token required)"}, status=401)
                    return
                selected_cfg = body.get("config") if isinstance(body, dict) else None
                question_set = str(body.get("question_set") or "golden_only") if isinstance(body, dict) else "golden_only"
                doc_id = (body.get("doc_id") or None) if isinstance(body, dict) else None
                self._send_json(
                    run_evaluation_suite(
                        selected_config=selected_cfg,
                        question_set=question_set,
                        doc_id=doc_id,
                    )
                )
                return

            self._send_json({"detail": f"Unknown route: {path}"}, status=404)
        except ValueError as exc:
            self._send_json({"detail": str(exc)}, status=400)
        except Exception:
            err = friendly_error_payload()
            logger.exception("Unhandled error in POST %s request_id=%s", path, err["request_id"])
            self._send_json(err, status=500)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        ingest_documents(load_corpus_documents())
    except Exception:
        pass
    server = ThreadingHTTPServer((host, port), RAGWebRequestHandler)
    print(f"RAG Service & Eval Harness Web UI listening at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the RAG Service & Eval Harness Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
