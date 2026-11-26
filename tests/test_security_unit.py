"""Offline unit tests for app.security helpers and sandbox guardrails."""

import unittest

from app.security import (
    SANDBOX_MAX_CHUNKS_PER_SESSION,
    SANDBOX_MAX_DOCS_PER_SESSION,
    SANDBOX_MAX_FILE_BYTES,
    create_visitor_session_id,
    create_visitor_session_value,
    parse_cookies,
    sanitize_source_label,
    strip_control_chars,
    validate_sandbox_upload,
    verify_visitor_session_value,
)


class TestSecurityUnit(unittest.TestCase):
    def test_strip_control_chars_preserves_tabs_and_newlines(self) -> None:
        raw = "hello\x00world\x07\tline1\nline2\x1f\x7f"
        self.assertEqual(strip_control_chars(raw), "helloworld\tline1\nline2")

    def test_sanitize_source_label_strips_internal_paths(self) -> None:
        self.assertEqual(
            sanitize_source_label("/Users/alice/docs/guide.md", title="Guide"),
            "Guide",
        )
        self.assertEqual(
            sanitize_source_label("data/corpus/01_pgvector.md"),
            "01_pgvector.md",
        )
        self.assertEqual(
            sanitize_source_label(r"C:\repo\secret\notes.txt", title="Notes"),
            "Notes",
        )
        self.assertEqual(sanitize_source_label("public-doc.md"), "public-doc.md")

    def test_parse_cookies_handles_empty_and_equals_in_value(self) -> None:
        self.assertEqual(parse_cookies(None), {})
        self.assertEqual(parse_cookies(""), {})
        parsed = parse_cookies("rag_visitor_session=abc.def=; theme=dark")
        self.assertEqual(
            parsed,
            {"rag_visitor_session": "abc.def=", "theme": "dark"},
        )

    def test_visitor_session_token_roundtrip_and_tamper_rejection(self) -> None:
        sid = create_visitor_session_id()
        token = create_visitor_session_value(sid, ttl_seconds=60)
        self.assertEqual(verify_visitor_session_value(token), sid)

        # Tamper signature
        payload, sig = token.split(".", 1)
        tampered_sig = ("0" if sig[0] != "0" else "1") + sig[1:]
        self.assertIsNone(verify_visitor_session_value(f"{payload}.{tampered_sig}"))

        # Expired token
        expired = create_visitor_session_value(sid, ttl_seconds=-10)
        self.assertIsNone(verify_visitor_session_value(expired))

        # Invalid session ID prefix
        bad_sid_token = create_visitor_session_value("admin_123", ttl_seconds=60)
        self.assertIsNone(verify_visitor_session_value(bad_sid_token))

    def test_validate_sandbox_upload_enforces_limits_and_extensions(self) -> None:
        ok, msg = validate_sandbox_upload("notes.md", "Hello RAG", 0, 0)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

        # Disallowed extension
        ok, _ = validate_sandbox_upload("script.py", "print(1)", 0, 0)
        self.assertFalse(ok)

        # Empty content
        ok, _ = validate_sandbox_upload("notes.txt", "", 0, 0)
        self.assertFalse(ok)

        # Exceeds byte limit
        oversized = "a" * (SANDBOX_MAX_FILE_BYTES + 1)
        ok, _ = validate_sandbox_upload("big.md", oversized, 0, 0)
        self.assertFalse(ok)

        # Exceeds doc limit
        ok, _ = validate_sandbox_upload(
            "notes.md", "ok", SANDBOX_MAX_DOCS_PER_SESSION, 0
        )
        self.assertFalse(ok)

        # Exceeds chunk limit
        ok, _ = validate_sandbox_upload(
            "notes.md", "ok", 0, SANDBOX_MAX_CHUNKS_PER_SESSION
        )
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
