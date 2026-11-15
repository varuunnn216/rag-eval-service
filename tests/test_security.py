"""Security & secrets verification tests against BASE_URL (default http://127.0.0.1:8000)."""

from __future__ import annotations

import json
import os
import re
import unittest
import urllib.error
import urllib.request

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")

SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"AQ\.[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"postgres(?:ql)?://[^:\s]+:[^@\s]+@"),
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(r"\bADMIN_TOKEN\b"),
]

INTERNAL_PATH_PATTERNS = [
    re.compile(r"/Users/[^\s\"']+"),
    re.compile(r"/var/task/[^\s\"']+"),
    re.compile(r"data/corpus/[^\s\"']+"),
]


def _server_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _fetch(path: str, method: str = "GET", body: dict | None = None, headers: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req_headers = dict(headers or {})
    if body is not None:
        req_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, dict(r.headers), r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8", errors="replace")


@unittest.skipUnless(_server_reachable(), f"Server not reachable at {BASE_URL}")
class TestLiveSecurity(unittest.TestCase):
    def test_1_served_static_assets_contain_no_secrets_or_internal_paths(self) -> None:
        status, _, html = _fetch("/")
        self.assertEqual(status, 200)
        assets = ["/"] + re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
        for asset in assets:
            st, _, content = _fetch(asset)
            self.assertEqual(st, 200, f"Failed fetching {asset}")
            for pat in SECRET_PATTERNS:
                self.assertIsNone(pat.search(content), f"Secret pattern {pat.pattern} found in {asset}")
            for pat in INTERNAL_PATH_PATTERNS:
                self.assertIsNone(pat.search(content), f"Internal path {pat.pattern} found in {asset}")

    def test_2_api_responses_redact_paths_traces_and_system_prompt_in_demo_mode(self) -> None:
        _, _, health_raw = _fetch("/health")
        health = json.loads(health_raw)
        demo_mode = bool(health.get("demo_mode"))

        st_docs, _, docs_raw = _fetch("/api/v1/documents")
        self.assertEqual(st_docs, 200)

        st_ask, _, ask_raw = _fetch(
            "/api/v1/ask",
            method="POST",
            body={"question": "Which bus route goes from Downtown to Lakeview?"},
        )
        self.assertEqual(st_ask, 200)

        st_eval, _, eval_raw = _fetch("/api/v1/eval/report")
        self.assertEqual(st_eval, 200)

        for label, raw in [("documents", docs_raw), ("ask", ask_raw), ("eval", eval_raw)]:
            for pat in SECRET_PATTERNS:
                self.assertIsNone(pat.search(raw), f"Secret leaked in {label}")
            self.assertNotIn("Traceback (most recent call last)", raw)
            if demo_mode:
                for pat in INTERNAL_PATH_PATTERNS:
                    self.assertIsNone(pat.search(raw), f"Internal path {pat.pattern} leaked in {label}")

        if demo_mode:
            ask_data = json.loads(ask_raw)
            self.assertIsNone(ask_data.get("prompt_sent"), "System prompt must not be exposed in DEMO_MODE")

    def test_3_api_docs_disabled_in_demo_mode(self) -> None:
        _, _, health_raw = _fetch("/health")
        if not json.loads(health_raw).get("demo_mode"):
            self.skipTest("Not in DEMO_MODE")
        for doc_path in ("/docs", "/redoc", "/openapi.json"):
            st, _, _ = _fetch(doc_path)
            self.assertEqual(st, 404, f"{doc_path} must be disabled in DEMO_MODE")

    def test_4_admin_endpoints_require_token_in_demo_mode(self) -> None:
        _, _, health_raw = _fetch("/health")
        if not json.loads(health_raw).get("demo_mode"):
            self.skipTest("Not in DEMO_MODE")
        checks = [
            ("POST", "/api/v1/eval/run", {}),
            ("POST", "/api/v1/eval/generate", {"doc_id": "doc_08_bluebird_cafe"}),
            ("POST", "/api/v1/ingest", {"documents": [{"id": "x", "title": "x", "content": "hello world"}]}),
            ("DELETE", "/api/v1/documents/doc_08_bluebird_cafe", None),
        ]
        for method, path, body in checks:
            st, _, _ = _fetch(path, method=method, body=body)
            self.assertIn(st, (401, 403), f"{method} {path} returned {st} without ADMIN_TOKEN")

    def test_5_security_headers_present(self) -> None:
        _, hdrs, _ = _fetch("/health")
        lower = {k.lower(): v for k, v in hdrs.items()}
        self.assertIn("content-security-policy", lower)
        self.assertEqual(lower.get("x-content-type-options"), "nosniff")
        self.assertEqual(lower.get("x-frame-options"), "DENY")
        self.assertIn("referrer-policy", lower)


if __name__ == "__main__":
    unittest.main()
