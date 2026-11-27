"""Unit tests for deterministic query expansion and synonym stem groups."""

import unittest

from app.retrieval.query_expansion import expand_query, get_synonym_stems


class TestQueryExpansion(unittest.TestCase):
    def test_empty_and_no_match_queries_return_empty_list(self) -> None:
        self.assertEqual(expand_query(""), [])
        self.assertEqual(expand_query("   "), [])
        self.assertEqual(expand_query("HNSW ef_construction parameter"), [])

    def test_expands_vocabulary_mismatched_terms_up_to_max_rewrites(self) -> None:
        rewrites = expand_query("Why is the light flashing?", max_rewrites=3)
        self.assertGreaterEqual(len(rewrites), 1)
        self.assertLessEqual(len(rewrites), 3)
        joined = " ".join(rewrites).lower()
        self.assertIn("blinking", joined)
        self.assertIn("led indicator", joined)
        # Original query should not be duplicated in output
        self.assertNotIn("why is the light flashing?", [r.lower() for r in rewrites])

    def test_get_synonym_stems_known_and_unknown(self) -> None:
        stems = get_synonym_stems("flash")
        self.assertEqual(stems, {"flash", "blink"})
        self.assertEqual(get_synonym_stems("pgvector"), {"pgvector"})


if __name__ == "__main__":
    unittest.main()
