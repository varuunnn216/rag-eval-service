from __future__ import annotations

import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import get_settings
from app.security import ADMIN_COOKIE_NAME, reset_rate_limits
from app.ui_server import RAGWebRequestHandler


class _FakeHandler(RAGWebRequestHandler):
    """In-process harness around RAGWebRequestHandler without opening TCP sockets."""

    def __init__(self, method: str, path: str, body: dict | None = None, headers: dict[str, str] | None = None):
        self.command = method
        self.path = path
        self.request_version = "HTTP/1.1"
        self.client_address = ("127.0.0.1", 54321)
        raw_body = json.dumps(body).encode("utf-8") if body is not None else b""
        req_headers = dict(headers or {})
        if body is not None:
            req_headers.setdefault("Content-Type", "application/json")
            req_headers.setdefault("Content-Length", str(len(raw_body)))
        self.headers = req_headers
        self.rfile = io.BytesIO(raw_body)
        self.wfile = io.BytesIO()
        self.status_code = 0
        self.response_headers: list[tuple[str, str]] = []

    def send_response(self, code: int, message: str | None = None) -> None:
        self.status_code = code

    def send_header(self, keyword: str, value: str) -> None:
        self.response_headers.append((keyword, value))

    def end_headers(self) -> None:
        pass

    def log_message(self, format: str, *args) -> None:
        pass


class AdminDemoModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_demo = os.environ.get("DEMO_MODE")
        self._orig_token = os.environ.get("ADMIN_TOKEN")
        os.environ["DEMO_MODE"] = "true"
        os.environ["ADMIN_TOKEN"] = "unit-test-admin-secret"
        get_settings.cache_clear()
        reset_rate_limits()

    def tearDown(self) -> None:
        if self._orig_demo is None:
            os.environ.pop("DEMO_MODE", None)
        else:
            os.environ["DEMO_MODE"] = self._orig_demo
        if self._orig_token is None:
            os.environ.pop("ADMIN_TOKEN", None)
        else:
            os.environ["ADMIN_TOKEN"] = self._orig_token
        get_settings.cache_clear()
        reset_rate_limits()

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, list[tuple[str, str]], dict | list]:
        h = _FakeHandler(method, path, body=body, headers=headers)
        if method == "GET":
            h.do_GET()
        elif method == "POST":
            h.do_POST()
        elif method == "DELETE":
            h.do_DELETE()
        raw = h.wfile.getvalue().decode("utf-8")
        parsed = json.loads(raw) if raw else {}
        return h.status_code, h.response_headers, parsed

    def test_admin_endpoints_return_401_without_session_or_token(self) -> None:
        status, _, _ = self._request(
            "POST",
            "/api/v1/ingest",
            {"documents": [{"title": "T", "content": "Hello world"}]},
        )
        self.assertEqual(status, 401)

        status, _, _ = self._request("DELETE", "/api/v1/documents/doc_01")
        self.assertEqual(status, 401)

        status, _, _ = self._request("POST", "/api/v1/eval/generate", {"doc_id": "doc_01"})
        self.assertEqual(status, 401)

        status, _, _ = self._request("POST", "/api/v1/eval/run", {})
        self.assertEqual(status, 401)

    def test_wrong_token_401_and_rate_limit_429(self) -> None:
        for _ in range(5):
            status, _, _ = self._request("POST", "/api/v1/admin/login", {"token": "wrong-token"})
            self.assertEqual(status, 401)

        status, _, _ = self._request("POST", "/api/v1/admin/login", {"token": "wrong-token"})
        self.assertEqual(status, 429)

    def test_right_token_sets_httponly_cookie_and_allows_ingest_and_hides_source_for_visitors(self) -> None:
        status, headers, payload = self._request(
            "POST",
            "/api/v1/admin/login",
            {"token": "unit-test-admin-secret"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("is_admin"))

        set_cookies = [v for k, v in headers if k.lower() == "set-cookie"]
        self.assertTrue(set_cookies)
        set_cookie = set_cookies[0]
        self.assertIn(f"{ADMIN_COOKIE_NAME}=", set_cookie)
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=Strict", set_cookie)
        cookie_pair = set_cookie.split(";", 1)[0]

        fake_doc = {
            "id": "doc_test_admin",
            "title": "Test Admin Doc",
            "source": "data/corpus/secret_path.md",
            "chunk_count": 3,
        }
        from types import SimpleNamespace
        fake_ing = SimpleNamespace(
            to_dict=lambda: {
                "documents_ingested": 1,
                "chunks_created": 3,
                "document_ids": ["doc_test_admin"],
            }
        )
        with patch("app.ui_server.ingest_documents", return_value=fake_ing), \
             patch("app.ui_server.list_documents", return_value=[fake_doc]):
            status, _, ing_data = self._request(
                "POST",
                "/api/v1/ingest",
                {
                    "documents": [
                        {
                            "id": "doc_test_admin",
                            "title": "Test Admin Doc",
                            "source": "data/corpus/secret_path.md",
                            "content": "# Test Admin Doc\n\nContent for test.",
                        }
                    ]
                },
                headers={"Cookie": cookie_pair},
            )
            self.assertEqual(status, 200)
            self.assertEqual(ing_data["chunks_created"], 3)

            status, _, docs_admin = self._request(
                "GET", "/api/v1/documents", headers={"Cookie": cookie_pair}
            )
            self.assertEqual(status, 200)
            self.assertEqual(docs_admin[0]["source"], "data/corpus/secret_path.md")

            status, _, docs_visitor = self._request("GET", "/api/v1/documents")
            self.assertEqual(status, 200)
            self.assertEqual(docs_visitor[0]["source"], "")
            self.assertEqual(docs_visitor[0]["chunk_count"], 3)

    def test_ui_error_styles_and_demo_banner_present(self) -> None:
        css = Path("app/static/styles.css").read_text(encoding="utf-8")
        self.assertIn(".ingest-feedback.is-error", css)
        self.assertIn(".toast-item.is-error", css)
        self.assertIn("var(--danger)", css)

        html = Path("app/static/index.html").read_text(encoding="utf-8")
        self.assertIn('id="docs-demo-banner"', html)
        self.assertIn('id="admin-login-link"', html)
        self.assertIn("Public demo: the document library is read-only.", html)


if __name__ == "__main__":
    unittest.main()
