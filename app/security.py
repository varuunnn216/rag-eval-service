from __future__ import annotations

import base64
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
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
ADMIN_COOKIE_NAME = "rag_admin_session"
ADMIN_SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours

VISITOR_COOKIE_NAME = "rag_visitor_session"
VISITOR_SESSION_TTL_SECONDS = 24 * 60 * 60  # 1 day

SANDBOX_MAX_DOCS_PER_SESSION = 3
SANDBOX_MAX_FILE_BYTES = 100 * 1024  # 100 KB
SANDBOX_MAX_CHUNKS_PER_SESSION = 20
SANDBOX_ALLOWED_EXTENSIONS = {".md", ".markdown", ".txt"}

_rate_lock = threading.Lock()
_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)
_login_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)
_ingest_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)
_ask_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)
_daily_counts: Dict[str, Tuple[str, int]] = {}  # key -> (YYYY-MM-DD, count)


def check_rate_limit(client_ip: str, limit: int = 120, window_seconds: int = 60) -> bool:
    """Sliding-window per-IP rate limiter. Returns True if allowed."""
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


def check_login_rate_limit(client_ip: str, limit: int = 5, window_seconds: int = 15 * 60) -> bool:
    """Rate limit admin login attempts (5 per 15 minutes per IP)."""
    now = time.monotonic()
    cutoff = now - window_seconds
    key = client_ip or "unknown"
    with _rate_lock:
        dq = _login_rate_buckets[key]
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            return False
        dq.append(now)
    return True


def reset_rate_limits() -> None:
    """Clear in-memory rate limit buckets and daily caps (for unit tests)."""
    with _rate_lock:
        _rate_buckets.clear()
        _login_rate_buckets.clear()
        _ingest_rate_buckets.clear()
        _ask_rate_buckets.clear()
        _daily_counts.clear()


def _check_bucket_limit(
    buckets: Dict[str, Deque[float]],
    client_ip: str,
    limit: int,
    window_seconds: int,
) -> bool:
    now = time.monotonic()
    cutoff = now - window_seconds
    key = client_ip or "unknown"
    with _rate_lock:
        dq = buckets[key]
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            return False
        dq.append(now)
    return True


def check_ingest_rate_limit(client_ip: str, limit: int = 5, window_seconds: int = 3600) -> bool:
    """Rate limit visitor sandbox uploads (5 per hour per IP)."""
    return _check_bucket_limit(_ingest_rate_buckets, client_ip, limit=limit, window_seconds=window_seconds)


def check_ask_rate_limit(client_ip: str, limit: int = 20, window_seconds: int = 3600) -> bool:
    """Rate limit visitor Ask requests (20 per hour per IP)."""
    return _check_bucket_limit(_ask_rate_buckets, client_ip, limit=limit, window_seconds=window_seconds)


def _check_daily_cap(cap_key: str, default_limit: int, env_var: str) -> Tuple[bool, str]:
    raw_limit = os.getenv(env_var, "").strip()
    limit = int(raw_limit) if raw_limit.isdigit() else default_limit
    if limit <= 0:
        return True, ""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _rate_lock:
        cur_date, cur_count = _daily_counts.get(cap_key, (today, 0))
        if cur_date != today:
            cur_date, cur_count = today, 0
        if cur_count >= limit:
            return False, "Daily demo limit reached, try again tomorrow"
        _daily_counts[cap_key] = (cur_date, cur_count + 1)
    return True, ""


def check_daily_ingest_cap() -> Tuple[bool, str]:
    return _check_daily_cap("ingest", default_limit=100, env_var="DEMO_DAILY_INGEST_CAP")


def check_daily_llm_cap() -> Tuple[bool, str]:
    return _check_daily_cap("llm", default_limit=500, env_var="DEMO_DAILY_LLM_CAP")


def _signing_key() -> bytes:
    token = (get_settings().admin_token or "").strip()
    return hashlib.sha256(f"rag-admin-session:{token}".encode("utf-8")).digest()


def verify_admin_token(candidate: str) -> bool:
    """Constant-time check of candidate token against ADMIN_TOKEN."""
    expected = (get_settings().admin_token or "").strip()
    cand = (candidate or "").strip()
    if not expected or not cand:
        return False
    return hmac.compare_digest(cand, expected)


def create_admin_session_value() -> str:
    """Create a signed, expiring admin session token (never embeds ADMIN_TOKEN)."""
    exp = int(time.time()) + ADMIN_SESSION_TTL_SECONDS
    payload = base64.urlsafe_b64encode(
        json.dumps({"role": "admin", "exp": exp}, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    sig = hmac.new(_signing_key(), payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_admin_session_value(cookie_val: Optional[str]) -> bool:
    """Verify signed admin session cookie and expiration."""
    if not cookie_val or "." not in cookie_val:
        return False
    expected_token = (get_settings().admin_token or "").strip()
    if not expected_token:
        return False
    payload_b64, sig = cookie_val.rsplit(".", 1)
    expected_sig = hmac.new(_signing_key(), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return False
    try:
        raw = base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8")
        data = json.loads(raw)
        return data.get("role") == "admin" and int(data.get("exp", 0)) > int(time.time())
    except Exception:
        return False


def parse_cookies(cookie_header: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not cookie_header:
        return out
    for part in cookie_header.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def is_admin_authenticated(headers: Any, cookies: Optional[Dict[str, str]] = None) -> bool:
    """Return True when not in DEMO_MODE, or when a valid X-Admin-Token / session cookie is present."""
    settings = get_settings()
    if not settings.demo_mode:
        return True
    hdr_token = (headers.get("X-Admin-Token") or headers.get("x-admin-token") or "").strip()
    auth_hdr = (headers.get("Authorization") or headers.get("authorization") or "").strip()
    if auth_hdr.lower().startswith("bearer "):
        auth_hdr = auth_hdr[7:].strip()
    if verify_admin_token(hdr_token) or verify_admin_token(auth_hdr):
        return True
    if cookies is None:
        cookies = parse_cookies(headers.get("Cookie") or headers.get("cookie"))
    return verify_admin_session_value(cookies.get(ADMIN_COOKIE_NAME))


def build_session_cookie_header(value: str, max_age: int = ADMIN_SESSION_TTL_SECONDS, secure: bool = True) -> str:
    parts = [
        f"{ADMIN_COOKIE_NAME}={value}",
        "Path=/",
        f"Max-Age={max_age}",
        "HttpOnly",
        "SameSite=Strict",
    ]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


def create_visitor_session_id() -> str:
    return f"sess_{secrets.token_urlsafe(18)}"


def create_visitor_session_value(session_id: str, ttl_seconds: int = VISITOR_SESSION_TTL_SECONDS) -> str:
    exp = int(time.time()) + ttl_seconds
    payload = json.dumps({"sid": session_id, "exp": exp}, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    sig = hmac.new(_signing_key(), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_visitor_session_value(cookie_val: Optional[str]) -> Optional[str]:
    if not cookie_val or "." not in cookie_val:
        return None
    payload_b64, sig = cookie_val.split(".", 1)
    expected_sig = hmac.new(_signing_key(), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    try:
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        if int(data.get("exp", 0)) <= int(time.time()):
            return None
        sid = str(data.get("sid") or "")
        return sid if sid.startswith("sess_") else None
    except Exception:
        return None


def build_visitor_session_cookie_header(
    session_id: str,
    max_age: int = VISITOR_SESSION_TTL_SECONDS,
    secure: bool = True,
) -> str:
    value = create_visitor_session_value(session_id, ttl_seconds=max_age)
    parts = [
        f"{VISITOR_COOKIE_NAME}={value}",
        "Path=/",
        f"Max-Age={max_age}",
        "HttpOnly",
        "SameSite=Lax",
    ]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


def get_or_create_visitor_session(
    headers: Dict[str, str],
    cookies: Optional[Dict[str, str]] = None,
    is_https: bool = True,
) -> Tuple[str, Optional[str]]:
    """Return (visitor_session_id, optional_set_cookie_header)."""
    if cookies is None:
        cookies = parse_cookies(headers.get("Cookie") or headers.get("cookie"))
    existing = verify_visitor_session_value(cookies.get(VISITOR_COOKIE_NAME))
    if existing:
        return existing, None
    new_sid = create_visitor_session_id()
    return new_sid, build_visitor_session_cookie_header(new_sid, secure=is_https)


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_control_chars(text: str) -> str:
    """Remove ASCII control characters while preserving tabs and newlines."""
    return _CONTROL_CHAR_RE.sub("", text or "")


def validate_sandbox_upload(
    filename: Optional[str],
    content: str,
    current_doc_count: int,
    current_chunk_count: int,
) -> Tuple[bool, str]:
    """Validate visitor sandbox upload limits and allowed file types."""
    if filename:
        ext = Path(filename.strip()).suffix.lower()
        if ext and ext not in SANDBOX_ALLOWED_EXTENSIONS:
            return False, "Only .md and .txt files (or pasted text) are allowed in the sandbox."
    raw_bytes = len((content or "").encode("utf-8"))
    if raw_bytes == 0:
        return False, "Document content cannot be empty."
    if raw_bytes > SANDBOX_MAX_FILE_BYTES:
        return False, f"Document exceeds the {SANDBOX_MAX_FILE_BYTES // 1024} KB sandbox limit."
    if current_doc_count >= SANDBOX_MAX_DOCS_PER_SESSION:
        return False, f"Sandbox limit reached: maximum {SANDBOX_MAX_DOCS_PER_SESSION} documents per session."
    if current_chunk_count >= SANDBOX_MAX_CHUNKS_PER_SESSION:
        return False, f"Sandbox limit reached: maximum {SANDBOX_MAX_CHUNKS_PER_SESSION} chunks per session."
    return True, ""


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
    """Strip internal filesystem paths (e.g. data/corpus/, /Users/...) for non-admins in DEMO_MODE."""
    s = (source or "").strip()
    if not s:
        return title or ""
    if "data/corpus/" in s or s.startswith("/") or "\\" in s:
        name = Path(s).name
        return title or name
    return s


def sanitize_ask_payload_for_demo(payload: Dict[str, Any], is_admin: bool = False) -> Dict[str, Any]:
    """Redact system prompt, internal paths, and raw model error strings for non-admins in DEMO_MODE."""
    if not get_settings().demo_mode or is_admin:
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


def sanitize_documents_for_demo(docs: List[Dict[str, Any]], is_admin: bool = False) -> List[Dict[str, Any]]:
    """Hide internal file paths from document listings for non-admins in DEMO_MODE."""
    if not get_settings().demo_mode or is_admin:
        return docs
    out: List[Dict[str, Any]] = []
    for d in docs:
        dd = dict(d)
        dd["source"] = ""
        out.append(dd)
    return out


def friendly_error_payload() -> Dict[str, str]:
    req_id = uuid.uuid4().hex[:12]
    return {"detail": "Something went wrong, please try again", "request_id": req_id}
