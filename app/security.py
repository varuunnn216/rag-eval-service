from __future__ import annotations

from collections import defaultdict, deque
import os
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, List
import uuid

from app.config import get_settings

CSP_HEADER = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'"
)

SECURITY_HEADERS: Dict[str, str] = {
    "Content-Security-Policy": CSP_HEADER,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
}

MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024

_rate_lock = threading.Lock()
_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)


def check_rate_limit(client_ip: str, limit: int = 120, window_seconds: int = 60) -> bool:
    """Simple sliding-window per-IP rate limiter. Returns True if allowed."""
    now = time.monotonic()
    cutoff = now - window_seconds
    key = client_ip or "unknown"
    with _rate_lock:
        dq = _rate_buckets[key]
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            return False
        dq.append(now)
    return True


def get_allowed_origins() -> List[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip() and o.strip() != "*"]
    return [
        "https://rag-eval-service.vercel.app",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]


def sanitize_source_label(source: str, title: str = "") -> str:
    """Strip internal filesystem paths (e.g. data/corpus/, /Users/...) in DEMO_MODE."""
    s = (source or "").strip()
    if not s:
        return title or ""
    if "data/corpus/" in s or s.startswith("/") or "\\" in s:
        name = Path(s).name
        return title or name
    return s


def sanitize_ask_payload_for_demo(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Redact system prompt, internal paths, and raw model error strings when DEMO_MODE=true."""
    if not get_settings().demo_mode:
        return payload
    out = dict(payload)
    out["prompt_sent"] = None
    if out.get("error"):
        out["error"] = "Something went wrong, please try again"
    if isinstance(out.get("citations"), list):
        cits = []
        for c in out["citations"]:
            if isinstance(c, dict):
                cc = dict(c)
                cc["source"] = sanitize_source_label(str(cc.get("source") or ""), str(cc.get("document_title") or ""))
                cits.append(cc)
            else:
                cits.append(c)
        out["citations"] = cits
    if isinstance(out.get("retrieved_chunks"), list):
        chunks = []
        for ch in out["retrieved_chunks"]:
            if isinstance(ch, dict):
                cc = dict(ch)
                cc["source"] = sanitize_source_label(str(cc.get("source") or ""), str(cc.get("document_title") or ""))
                chunks.append(cc)
            else:
                chunks.append(ch)
        out["retrieved_chunks"] = chunks
    return out


def sanitize_documents_for_demo(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Redact internal file paths from document listings in DEMO_MODE."""
    if not get_settings().demo_mode:
        return docs
    out: List[Dict[str, Any]] = []
    for d in docs:
        dd = dict(d)
        dd["source"] = sanitize_source_label(str(dd.get("source") or ""), str(dd.get("title") or ""))
        out.append(dd)
    return out


def friendly_error_payload() -> Dict[str, str]:
    req_id = uuid.uuid4().hex[:12]
    return {"detail": "Something went wrong, please try again", "request_id": req_id}
