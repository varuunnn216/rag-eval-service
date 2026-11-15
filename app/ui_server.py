from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import time
from typing import Any, Dict
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
    delete_document,
    get_document_chunks,
    ingest_documents,
    list_documents,
)
from app.retrieval.hybrid import retrieve_candidates, retrieve_candidates_with_timing
from app.schemas import AskRequest, AskResponse, DocumentInput
from eval.runner import run_evaluation_suite

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _get_health_payload() -> Dict[str, Any]:
    settings = get_settings()
    docs = list_documents()
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
        "reranker_model": settings.reranker_model,
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

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"detail": "Not found"}, status=404)
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        if file_path.suffix == ".svg":
            mime = "image/svg+xml"
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_sse_event(self, payload: Dict[str, Any]) -> None:
        line = f"data: {json.dumps(payload, default=str)}\n\n".encode("utf-8")
        self.wfile.write(line)
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_file(STATIC_DIR / "index.html")
            return
        if path == "/favicon.ico" or path == "/favicon.svg":
            self._send_file(STATIC_DIR / "favicon.svg")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            target = (STATIC_DIR / rel).resolve()
            if str(target).startswith(str(STATIC_DIR.resolve())):
                self._send_file(target)
                return
            self._send_json({"detail": "Forbidden"}, status=403)
            return
        if path == "/health":
            self._send_json(_get_health_payload())
            return
        if path == "/api/v1/documents":
            self._send_json(list_documents())
            return
        if path.startswith("/api/v1/documents/") and path.endswith("/chunks"):
            parts = path.split("/")
            if len(parts) >= 5:
                doc_id = unquote(parts[4])
                self._send_json(get_document_chunks(doc_id))
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
        if path.startswith("/api/v1/documents/"):
            doc_id = unquote(path[len("/api/v1/documents/") :].strip("/"))
            if doc_id:
                self._send_json(delete_document(doc_id))
                return
        self._send_json({"detail": f"Unknown route: {path}"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._read_json_body()
            if path == "/api/v1/ask":
                req = _parse_ask_request(body)
                resp = answer_question(req)
                self._send_json(resp.to_dict())
                return

            if path == "/api/v1/ask/stream":
                req = _parse_ask_request(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
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
                resp = answer_question(req)
                # Stream answer tokens progressively for smooth visual feedback
                if not resp.abstained and not resp.error and resp.answer:
                    words = resp.answer.split(" ")
                    accum = []
                    step = max(2, len(words) // 8)
                    for i in range(0, len(words), step):
                        accum.extend(words[i : i + step])
                        self._send_sse_event({"stage": "delta", "text": " ".join(accum)})
                        time.sleep(0.018)
                self._send_sse_event({"stage": "done", "response": resp.to_dict()})
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
                results, latency_ms = retrieve_candidates(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    candidate_pool=candidate_pool,
                    use_reranker=use_reranker,
                    document_ids=doc_ids,
                    min_sparse_score=min_sparse,
                )
                self._send_json(
                    {
                        "query": query,
                        "mode": mode,
                        "use_reranker": use_reranker,
                        "results": [r.to_dict() for r in results],
                        "latency_ms": latency_ms,
                    }
                )
                return

            if path == "/api/v1/ingest":
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
                res = ingest_documents(docs)
                self._send_json(res.to_dict())
                return

            if path.startswith("/api/v1/documents/") and path.endswith("/delete"):
                parts = path.split("/")
                if len(parts) >= 5:
                    doc_id = unquote(parts[4])
                    self._send_json(delete_document(doc_id))
                    return

            if path == "/api/v1/eval/run":
                selected_cfg = body.get("config") if isinstance(body, dict) else None
                self._send_json(run_evaluation_suite(selected_config=selected_cfg))
                return

            self._send_json({"detail": f"Unknown route: {path}"}, status=404)
        except Exception as exc:
            self._send_json({"detail": str(exc)}, status=400)


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
