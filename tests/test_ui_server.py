import unittest

from app.ui_server import STATIC_DIR, _get_health_payload


class TestWebUIServer(unittest.TestCase):
    def test_static_assets_exist(self) -> None:
        self.assertTrue((STATIC_DIR / "index.html").exists())
        self.assertTrue((STATIC_DIR / "styles.css").exists())
        self.assertTrue((STATIC_DIR / "app.js").exists())

    def test_health_payload_reports_corpus_counts(self) -> None:
        health = _get_health_payload()
        self.assertEqual(health["status"], "ok")
        self.assertGreaterEqual(health["documents_count"], 6)
        self.assertGreaterEqual(health["chunks_count"], 25)


if __name__ == "__main__":
    unittest.main()
