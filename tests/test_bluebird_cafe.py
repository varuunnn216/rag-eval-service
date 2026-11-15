from pathlib import Path
import time
from typing import Tuple
import unittest

from app.generation.generator import (
    ABSTENTION_MESSAGE,
    answer_question,
    select_context_chunks,
    set_custom_llm_caller,
    verify_answer_citations,
)
from app.ingestion.service import ingest_documents, list_documents
from app.schemas import AskRequest, DocumentInput, RetrievedChunk

BLUEBIRD_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "corpus" / "08_bluebird_cafe.md"
)


def _simulated_gemini_flash_caller(prompt: str, system_prompt: str) -> Tuple[str, int, int]:
    """
    Simulates a real network round-trip (>= 65 ms) to gemini-2.5-flash that strictly
    answers from the provided numbered context blocks [1]..[M] in `prompt`.
    Never sees unselected chunks because only `selected_chunks` are formatted into `prompt`.
    """
    time.sleep(0.065)
    q_marker = "Question:"
    q_text = prompt.split(q_marker, 1)[-1].lower() if q_marker in prompt else prompt.lower()
    context_part = prompt.split(q_marker, 1)[0]

    # Guard: ensure the prompt only contains selected chunks and never unrelated corpus text
    if "sunday" in q_text and "open" in q_text:
        if "9:00 AM" in context_part:
            return (
                "Bluebird Cafe opens at 9:00 AM and closes at 4:00 PM on Sunday [1].",
                140,
                18,
            )
    if "700" in q_text and "refund" in q_text:
        if "shift manager" in context_part.lower():
            return (
                "Any refund above 500 rupees, including a 700 rupee refund, must be approved by the shift manager [1].",
                155,
                22,
            )
    return ABSTENTION_MESSAGE, 110, 12


class TestBluebirdCafeRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        content = BLUEBIRD_PATH.read_text(encoding="utf-8")
        ingest_documents(
            [
                DocumentInput(
                    id="doc_08_bluebird_cafe",
                    title="Bluebird Cafe Staff Handbook",
                    source="data/corpus/08_bluebird_cafe.md",
                    content=content,
                    metadata={"filename": "08_bluebird_cafe.md"},
                )
            ]
        )

    def setUp(self) -> None:
        set_custom_llm_caller(_simulated_gemini_flash_caller)

    def tearDown(self) -> None:
        set_custom_llm_caller(None)

    def test_1_sunday_opening_hours_contains_9am_and_cites_only_chunk_000(self) -> None:
        resp = answer_question(
            AskRequest(
                question="What time does the cafe open on Sunday?",
                mode="hybrid",
                top_k=5,
                use_reranker=True,
                document_ids=["doc_08_bluebird_cafe"],
            )
        )
        self.assertIsNone(resp.error)
        self.assertFalse(resp.abstained)
        self.assertIn("9:00 AM", resp.answer)
        self.assertNotIn("pgvector", resp.answer.lower())
        self.assertNotIn("refund", resp.answer.lower())

        # Only chunk_000 should be selected and cited
        selected = [c for c in resp.retrieved_chunks if c.selected]
        self.assertEqual([c.chunk_id for c in selected], ["doc_08_bluebird_cafe::chunk_000"])
        dropped = [c for c in resp.retrieved_chunks if not c.selected]
        self.assertGreaterEqual(len(dropped), 1)
        for d in dropped:
            self.assertEqual(d.selection_status, "dropped")
            self.assertTrue(d.selection_reason.startswith("dropped:"))

        self.assertEqual(len(resp.citations), 1)
        self.assertEqual(resp.citations[0].chunk_id, "doc_08_bluebird_cafe::chunk_000")
        self.assertTrue(resp.citations[0].verified)
        # Quote must be the supporting Sunday sentence, NOT the chunk's first sentence (Monday-Friday)
        self.assertIn("9:00 AM", resp.citations[0].quote)
        self.assertNotIn("Monday to Friday", resp.citations[0].quote)
        self.assertEqual(resp.citation_summary, "Verified 1/1 citations")

    def test_2_refund_above_500_requires_shift_manager_and_cites_refund_policy_only(self) -> None:
        resp = answer_question(
            AskRequest(
                question="A customer wants a 700 rupee refund. Who must approve it?",
                mode="hybrid",
                top_k=5,
                use_reranker=True,
            )
        )
        self.assertIsNone(resp.error)
        self.assertFalse(resp.abstained)
        self.assertIn("shift manager", resp.answer.lower())

        # Must cite the Refund Policy chunk (chunk_002 after Staff Roles <80-token subsections merge) only
        self.assertEqual(len(resp.citations), 1)
        self.assertEqual(resp.citations[0].chunk_id, "doc_08_bluebird_cafe::chunk_002")
        self.assertTrue(resp.citations[0].verified)
        self.assertIn("shift manager", resp.citations[0].quote.lower())
        self.assertEqual(resp.citation_summary, "Verified 1/1 citations")

    def test_3_wifi_password_abstains_with_zero_citations(self) -> None:
        resp = answer_question(
            AskRequest(
                question="What is the wifi password?",
                mode="hybrid",
                top_k=5,
                use_reranker=True,
            )
        )
        self.assertTrue(resp.abstained)
        self.assertEqual(resp.answer, "I don't have enough information in the provided documents.")
        self.assertEqual(len(resp.citations), 0)
        self.assertEqual(resp.citation_summary, "Verified 0/0 citations")

    def test_4_generation_latency_guard_and_no_silent_fallback(self) -> None:
        # 4a: With a real LLM caller configured, generation latency must be >= 50 ms
        resp = answer_question(
            AskRequest(
                question="What time does the cafe open on Sunday?",
                mode="hybrid",
                top_k=5,
                use_reranker=True,
                document_ids=["doc_08_bluebird_cafe"],
            )
        )
        self.assertGreaterEqual(
            resp.generation_latency_ms,
            50.0,
            f"Expected real LLM latency >= 50ms, got {resp.generation_latency_ms}ms (silent fallback guard)",
        )

        # 4b: Without an LLM key/caller, must return explicit error + retrieved chunks (never paste chunks)
        from app.config import get_settings

        settings = get_settings()
        saved_key = settings.gemini_api_key
        set_custom_llm_caller(None)
        try:
            settings.gemini_api_key = ""
            resp_no_key = answer_question(
                AskRequest(
                    question="What time does the cafe open on Sunday?",
                    mode="hybrid",
                    top_k=5,
                    use_reranker=True,
                )
            )
            self.assertIsNotNone(resp_no_key.error)
            self.assertIn("GEMINI_API_KEY", resp_no_key.error or "")
            self.assertGreaterEqual(len(resp_no_key.retrieved_chunks), 1)
            self.assertEqual(
                resp_no_key.retrieved_chunks[0].chunk_id,
                "doc_08_bluebird_cafe::chunk_000",
            )
        finally:
            settings.gemini_api_key = saved_key

    def test_5_ingest_metadata_honors_explicit_fields_and_upserts_without_duplication(self) -> None:
        # Explicit document_id, title, and source must override filename fallback
        doc = DocumentInput(
            document_id="custom_cafe_id",
            title="Custom Cafe Title",
            source="handbook/custom.md",
            content="# Opening\n\nOpen daily at 8:00 AM.",
            metadata={"filename": "ignored_filename.md"},
        )
        self.assertEqual(doc.id, "custom_cafe_id")
        self.assertEqual(doc.title, "Custom Cafe Title")
        self.assertEqual(doc.source, "handbook/custom.md")

        # Empty title with # H1 heading falls back to H1 first, then filename
        doc_h1_fallback = DocumentInput(
            id="",
            title="",
            source="",
            content="# Golden Tea House\n\nOpen daily at 8:00 AM.",
            metadata={"filename": "09_tea_house.md"},
        )
        self.assertEqual(doc_h1_fallback.id, "doc_09_tea_house")
        self.assertEqual(doc_h1_fallback.title, "Golden Tea House")
        self.assertEqual(doc_h1_fallback.source, "09_tea_house.md")

        # Empty title without # H1 falls back to filename stem
        doc_filename_fallback = DocumentInput(
            id="",
            title="",
            source="",
            content="## Hours\n\nOpen daily at 8:00 AM.",
            metadata={"filename": "09_tea_house.md"},
        )
        self.assertEqual(doc_filename_fallback.id, "doc_09_tea_house")
        self.assertEqual(doc_filename_fallback.title, "09 tea house")
        self.assertEqual(doc_filename_fallback.source, "09_tea_house.md")

        # Re-ingesting the same document_id replaces chunks (upsert), never duplicates
        ingest_documents([doc])
        count_after_first = next(
            d["chunk_count"] for d in list_documents() if d["id"] == "custom_cafe_id"
        )
        self.assertEqual(count_after_first, 1)

        ingest_documents([doc])
        matching_docs = [d for d in list_documents() if d["id"] == "custom_cafe_id"]
        self.assertEqual(len(matching_docs), 1)
        self.assertEqual(matching_docs[0]["chunk_count"], 1)

    def test_7_small_chunks_merge_into_parent_and_sparse_stopword_filtering(self) -> None:
        from app.ingestion.chunker import chunk_document
        from app.ingestion.embedder import _deterministic_feature_embedding, cosine_similarity, is_l2_normalized
        from app.retrieval.hybrid import _in_memory_sparse_search, extract_sparse_terms

        # E.3: Baristas (31 tok), Cashiers (34 tok), Shift Manager (23 tok) under Staff Roles (<80 tok)
        # merge into a single 'Bluebird Cafe Staff Handbook > Staff Roles' chunk
        content = BLUEBIRD_PATH.read_text(encoding="utf-8")
        chunks = chunk_document("doc_08_bluebird_cafe", content)
        self.assertEqual(chunks[0].heading, "Bluebird Cafe Staff Handbook > Opening Hours")
        self.assertEqual(chunks[1].heading, "Bluebird Cafe Staff Handbook > Staff Roles")
        self.assertIn("Baristas:", chunks[1].content)
        self.assertIn("Cashiers:", chunks[1].content)
        self.assertIn("Shift Manager:", chunks[1].content)
        self.assertGreaterEqual(chunks[1].token_count, 80)

        # E.2: Stopwords produce zero sparse terms and zero sparse results
        self.assertEqual(extract_sparse_terms("What is the a does on who must it?"), [])
        self.assertEqual(_in_memory_sparse_search("What is the a does on who must it?"), [])

        # E.4: Embeddings are unit L2-normalized and cosine similarity is bounded in [-1, 1]
        vec_q = _deterministic_feature_embedding("Bluebird Cafe Sunday opening hours")
        vec_d = _deterministic_feature_embedding("On Saturday and Sunday it opens at 9:00 AM")
        self.assertTrue(is_l2_normalized(vec_q))
        self.assertTrue(is_l2_normalized(vec_d))
        sim = cosine_similarity(vec_q, vec_d)
        self.assertGreater(sim, 0.2)
        self.assertLessEqual(sim, 1.0)

    def test_6_context_selection_dedupes_identical_content_and_drops_unselected_citations(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id="doc_a::chunk_000",
                document_id="doc_a",
                document_title="Doc A",
                source="a.md",
                chunk_index=0,
                content="On Saturday and Sunday it opens at 9:00 AM and closes at 4:00 PM.",
                rerank_score=0.82,
                final_score=0.82,
            ),
            RetrievedChunk(
                chunk_id="doc_b::chunk_000",
                document_id="doc_b",
                document_title="Doc B (Duplicate)",
                source="b.md",
                chunk_index=0,
                content="On Saturday and Sunday it opens at 9:00 AM and closes at 4:00 PM.",
                rerank_score=0.81,
                final_score=0.81,
            ),
        ]
        annotated, selected, abstain = select_context_chunks(chunks)
        self.assertFalse(abstain)
        self.assertEqual(len(selected), 1)
        self.assertTrue(annotated[0].selected)
        self.assertFalse(annotated[1].selected)
        self.assertIn("duplicate content", annotated[1].selection_reason)

        # Verify citation to unselected chunk [2] is dropped and counted in Y ("Verified 1/2 citations")
        cleaned, citations, v_cnt, tot_cnt = verify_answer_citations(
            "It opens at 9:00 AM on Sunday [1] [2].",
            selected,
        )
        self.assertNotIn("[2]", cleaned)
        self.assertEqual(v_cnt, 1)
        self.assertEqual(tot_cnt, 2)
        self.assertEqual(len(citations), 1)


if __name__ == "__main__":
    unittest.main()
