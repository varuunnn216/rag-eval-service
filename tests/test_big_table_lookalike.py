import os
import unittest
from pathlib import Path

from app.generation.generator import (
    ABSTENTION_MESSAGE,
    answer_question,
    check_lexical_evidence,
    set_custom_llm_caller,
)
from app.ingestion.chunker import chunk_document, estimate_token_count, linearize_tables
from app.ingestion.service import ingest_documents
from app.retrieval.reranker import truncate_passage_for_reranker
from app.schemas import AskRequest, DocumentInput, RetrievedChunk
from eval.runner import _grounded_eval_synthesizer


class TestBigTableAndLookalike(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["DISABLE_NEURAL_RERANKER"] = "1"
        os.environ["DISABLE_GEMINI_EMBEDDINGS"] = "1"
        md_path = Path(__file__).resolve().parent.parent / "data" / "corpus" / "14_meridian_transit_handbook.md"
        cls.meridian_md = md_path.read_text(encoding="utf-8")
        ingest_documents(
            [
                DocumentInput(
                    id="doc_14_meridian_transit",
                    title="Meridian City Transit Authority Rider Handbook, 2026 Edition",
                    source="14_meridian_transit_handbook.md",
                    content=cls.meridian_md,
                )
            ]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("DISABLE_NEURAL_RERANKER", None)
        os.environ.pop("DISABLE_GEMINI_EMBEDDINGS", None)

    def tearDown(self) -> None:
        set_custom_llm_caller(None)

    def test_large_table_splits_into_row_groups_under_250_tokens(self) -> None:
        chunks = chunk_document("doc_14_meridian_transit", self.meridian_md)
        route_chunks = [c for c in chunks if "Selected Routes" in c.heading and "| Route |" in c.content]
        self.assertGreaterEqual(len(route_chunks), 2)
        for c in route_chunks:
            self.assertLessEqual(c.token_count, 250)
            self.assertIn("| Route |", c.content)
            lin = linearize_tables(c.content)
            self.assertIn("Columns:", lin)
            self.assertIn("Route:", lin)

    def test_reranker_truncates_passage_with_warning_and_keeps_heading(self) -> None:
        long_ctx = "Meridian > Bus > Selected Routes\n" + ("word " * 500)
        with self.assertLogs("app.retrieval.reranker", level="WARNING") as cm:
            truncated = truncate_passage_for_reranker(long_ctx, chunk_id="c1", max_chars=200)
        self.assertLessEqual(len(truncated), 200)
        self.assertTrue(truncated.startswith("Meridian > Bus > Selected Routes"))
        self.assertTrue(any("Truncated reranker passage" in msg for msg in cm.output))

    def test_lexical_evidence_check_two_signal(self) -> None:
        c_match = RetrievedChunk(
            chunk_id="c1",
            document_id="doc_14_meridian_transit",
            document_title="Meridian",
            source="14_meridian_transit_handbook.md",
            chunk_index=0,
            heading="Bus > Selected Routes",
            content="Route: 41. Type: Local. From and to: Downtown to Lakeview.",
            token_count=20,
            sparse_rank=1,
            sparse_score=0.8,
            rrf_score=0.03,
        )
        c_other = RetrievedChunk(
            chunk_id="c2",
            document_id="doc_14_meridian_transit",
            document_title="Meridian",
            source="14_meridian_transit_handbook.md",
            chunk_index=1,
            heading="Fares",
            content="Bus fare is 2.00 USD.",
            token_count=10,
            sparse_rank=2,
            sparse_score=0.2,
            rrf_score=0.02,
        )
        self.assertTrue(check_lexical_evidence("Which bus route goes from Downtown to Lakeview?", [c_match, c_other]))
        self.assertFalse(check_lexical_evidence("Can I sell my vegetables at a market?", [c_match, c_other]))

    def test_meridian_big_table_lookalike_queries_and_traps(self) -> None:
        set_custom_llm_caller(_grounded_eval_synthesizer)
        answerable_cases = [
            ("Which bus route goes from Downtown to Lakeview?", "41"),
            ("Which bus route goes from Downtown to Lakeside?", "14"),
            ("When does Route 22 stop running at night?", "10:30 PM"),
            ("How often does the E5 express run?", "every 20 min"),
        ]
        for q, expected_substr in answerable_cases:
            resp = answer_question(
                AskRequest(question=q, document_ids=["doc_14_meridian_transit"], use_reranker=True)
            )
            self.assertFalse(resp.abstained, f"False abstention on {q!r}: {resp.abstain_reason}")
            self.assertIn(expected_substr.lower(), resp.answer.lower(), f"Missing {expected_substr!r} in {resp.answer!r}")
            self.assertGreaterEqual(len(resp.citations), 1)

        # Bike-share membership is unanswerable in Meridian handbook; other 3 are unanswerable across corpus
        r_bike = answer_question(
            AskRequest(
                question="How much does a bike-share membership cost?",
                document_ids=["doc_14_meridian_transit"],
                use_reranker=True,
            )
        )
        self.assertTrue(r_bike.abstained)
        self.assertEqual(r_bike.citations, [])

        for trap_q, doc_filter in [
            ("Does the ZX-400 support Wi-Fi 7?", None),
            ("Can I sell my vegetables at a market?", None),
            ("What is the Wi-Fi password?", ["doc_14_meridian_transit"]),
        ]:
            resp = answer_question(AskRequest(question=trap_q, document_ids=doc_filter, use_reranker=True))
            self.assertTrue(resp.abstained, f"Failed to abstain on trap {trap_q!r}: {resp.answer!r}")
            self.assertEqual(resp.citations, [])

    def test_lexical_fallback_abstain_reason_when_llm_abstains(self) -> None:
        set_custom_llm_caller(lambda _p, _s: (ABSTENTION_MESSAGE, 50, 10))
        resp = answer_question(
            AskRequest(
                question="Which bus route goes from Downtown to Lakeview?",
                document_ids=["doc_14_meridian_transit"],
                rerank_min_score=0.99,  # force rerank below min so lexical fallback triggers
            )
        )
        self.assertTrue(resp.abstained)
        self.assertEqual(resp.fallback_used, "lexical_evidence")
        self.assertIn("Checked", resp.abstain_reason or "")
        self.assertIn("keyword evidence; the model found no supporting answer", resp.abstain_reason or "")


if __name__ == "__main__":
    unittest.main()
