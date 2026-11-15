from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import json
import os
import unittest
from pathlib import Path

from app.config import get_settings
from app.ingestion.service import cleanup_expired_sandbox_documents, ingest_documents
from app.retrieval.hybrid import _extra_ingested_docs, delete_in_memory_document
from app.schemas import DocumentInput
from app.security import VISITOR_COOKIE_NAME, reset_rate_limits
from app.ui_server import RAGWebRequestHandler


class _FakeHandler(RAGWebRequestHandler):
    def __init__(self, method: str, path: str, body: dict | None = None, headers: dict[str, str] | None = None):
        self.command = method
        self.path = path
        self.request_version = "HTTP/1.1"
        self.client_address = ("127.0.0.1", 54322)
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


class VisitorSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_demo = os.environ.get("DEMO_MODE")
        self._orig_token = os.environ.get("ADMIN_TOKEN")
        os.environ["DEMO_MODE"] = "true"
        os.environ["ADMIN_TOKEN"] = "sandbox-test-admin"
        get_settings.cache_clear()
        reset_rate_limits()
        self._created_doc_ids: list[str] = []

    def tearDown(self) -> None:
        for did in self._created_doc_ids:
            delete_in_memory_document(did)
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
        try:
            parsed = json.loads(raw) if raw else {}
        except Exception:
            parsed = {}
        return h.status_code, h.response_headers, parsed

    def _extract_visitor_cookie(self, headers: list[tuple[str, str]]) -> str:
        for k, v in headers:
            if k.lower() == "set-cookie" and v.startswith(f"{VISITOR_COOKIE_NAME}="):
                return v.split(";", 1)[0]
        return ""

    def test_two_sessions_strict_isolation_and_shared_library_protected(self) -> None:
        # Session A gets cookie & uploads a private doc
        st, hdrs_a, _ = self._request("GET", "/api/v1/documents")
        self.assertEqual(st, 200)
        cookie_a = self._extract_visitor_cookie(hdrs_a)
        self.assertTrue(cookie_a.startswith(f"{VISITOR_COOKIE_NAME}="))

        st_ing, _, ing_a = self._request(
            "POST",
            "/api/v1/ingest",
            {
                "documents": [
                    {
                        "title": "Visitor A Secret Notes",
                        "source": "secret_a.md",
                        "content": "# Secret A\n\nThe secret code for Visitor A is ZEBRA-991.\x07",
                    }
                ]
            },
            headers={"Cookie": cookie_a},
        )
        self.assertEqual(st_ing, 200)
        doc_id_a = ing_a["document_ids"][0]
        self._created_doc_ids.append(doc_id_a)

        # Session A sees its doc and can view chunks (control char \x07 stripped)
        st, _, docs_a = self._request("GET", "/api/v1/documents", headers={"Cookie": cookie_a})
        self.assertEqual(st, 200)
        self.assertTrue(any(d["id"] == doc_id_a and d.get("is_sandbox") for d in docs_a))
        st, _, chunks_a = self._request("GET", f"/api/v1/documents/{doc_id_a}/chunks", headers={"Cookie": cookie_a})
        self.assertEqual(st, 200)
        self.assertEqual(len(chunks_a), 1)
        self.assertNotIn("\x07", chunks_a[0]["content"])

        # Session B gets its own cookie and MUST NOT see Session A's doc, chunks, or search results
        st, hdrs_b, docs_b = self._request("GET", "/api/v1/documents")
        self.assertEqual(st, 200)
        cookie_b = self._extract_visitor_cookie(hdrs_b)
        self.assertNotEqual(cookie_a, cookie_b)
        self.assertFalse(any(d["id"] == doc_id_a for d in docs_b))

        st, _, chunks_b = self._request("GET", f"/api/v1/documents/{doc_id_a}/chunks", headers={"Cookie": cookie_b})
        self.assertEqual(st, 200)
        self.assertEqual(chunks_b, [])

        st_search_b, _, search_b = self._request(
            "POST",
            "/api/v1/search",
            {"query": "ZEBRA-991", "mode": "hybrid", "scope_mode": "all"},
            headers={"Cookie": cookie_b},
        )
        self.assertEqual(st_search_b, 200)
        self.assertFalse(any(r["document_id"] == doc_id_a for r in search_b["results"]))

        # Session B cannot delete Session A's doc or shared library doc
        st_del_ab, _, _ = self._request("DELETE", f"/api/v1/documents/{doc_id_a}", headers={"Cookie": cookie_b})
        self.assertEqual(st_del_ab, 404)
        st_del_shared, _, _ = self._request("DELETE", "/api/v1/documents/doc_pgvector_indexing", headers={"Cookie": cookie_b})
        self.assertEqual(st_del_shared, 401)

        # Session A CAN delete its own doc
        st_del_aa, _, del_aa = self._request("DELETE", f"/api/v1/documents/{doc_id_a}", headers={"Cookie": cookie_a})
        self.assertEqual(st_del_aa, 200)
        self.assertTrue(del_aa["deleted"])

    def test_sandbox_limits_file_types_and_size_checks(self) -> None:
        _, hdrs, _ = self._request("GET", "/api/v1/documents")
        cookie = self._extract_visitor_cookie(hdrs)

        # Disallowed file extension
        st, _, err = self._request(
            "POST",
            "/api/v1/ingest",
            {
                "documents": [
                    {
                        "title": "Bad Script",
                        "source": "evil.py",
                        "content": "print('hello')",
                        "metadata": {"filename": "evil.py"},
                    }
                ]
            },
            headers={"Cookie": cookie},
        )
        self.assertEqual(st, 400)
        self.assertIn(".md and .txt", err["detail"])

        # Oversized (> 100 KB)
        st, _, err = self._request(
            "POST",
            "/api/v1/ingest",
            {
                "documents": [
                    {
                        "title": "Too Big",
                        "source": "big.md",
                        "content": "a" * (105 * 1024),
                        "metadata": {"filename": "big.md"},
                    }
                ]
            },
            headers={"Cookie": cookie},
        )
        self.assertEqual(st, 400)
        self.assertIn("100 KB", err["detail"])

        reset_rate_limits()
        # Max 3 docs per session
        for i in range(3):
            st, _, res = self._request(
                "POST",
                "/api/v1/ingest",
                {
                    "documents": [
                        {
                            "title": f"Doc {i}",
                            "source": f"doc_{i}.md",
                            "content": f"# Doc {i}\n\nShort text {i}.",
                            "metadata": {"filename": f"doc_{i}.md"},
                        }
                    ]
                },
                headers={"Cookie": cookie},
            )
            self.assertEqual(st, 200)
            self._created_doc_ids.extend(res["document_ids"])

        st_4th, _, err_4th = self._request(
            "POST",
            "/api/v1/ingest",
            {
                "documents": [
                    {
                        "title": "Doc 4",
                        "source": "doc_3.md",
                        "content": "# Doc 4\n\nToo many.",
                        "metadata": {"filename": "doc_3.md"},
                    }
                ]
            },
            headers={"Cookie": cookie},
        )
        self.assertEqual(st_4th, 400)
        self.assertIn("maximum 3 documents", err_4th["detail"])

    def test_expired_sandbox_docs_are_cleaned_up(self) -> None:
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        doc = DocumentInput(
            id="sandbox_expired_doc",
            title="Expired Doc",
            source="expired.md",
            content="# Expired\n\nOld sandbox content.",
            session_id="sess_expired_test",
            expires_at=past,
        )
        ingest_documents([doc], session_id="sess_expired_test", expires_at=past)
        self._created_doc_ids.append("sandbox_expired_doc")

        removed = cleanup_expired_sandbox_documents()
        self.assertGreaterEqual(removed, 1)
        self.assertFalse(any(d.id == "sandbox_expired_doc" for d in _extra_ingested_docs))

    def test_self_hosted_fonts_and_no_google_fonts(self) -> None:
        html = Path("app/static/index.html").read_text(encoding="utf-8")
        self.assertNotIn("fonts.googleapis.com", html)
        self.assertNotIn("fonts.gstatic.com", html)
        self.assertIn("/static/fonts/SpaceGrotesk.woff2", html)
        self.assertIn("/static/fonts/JetBrainsMono.woff2", html)
        self.assertTrue(Path("app/static/fonts/SpaceGrotesk.woff2").is_file())
        self.assertTrue(Path("app/static/fonts/JetBrainsMono.woff2").is_file())


if __name__ == "__main__":
    unittest.main()
