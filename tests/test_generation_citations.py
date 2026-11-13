import unittest

from app.generation.generator import (
    extract_and_verify_citations,
    generate_grounded_answer,
)
from app.schemas import RetrievedChunk


class TestGenerationAndCitations(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            RetrievedChunk(
                chunk_id="doc_pg::chunk_001",
                document_id="doc_pg",
                document_title="pgvector HNSW Indexing",
                source="pgvector.md",
                chunk_index=1,
                heading="HNSW Build Parameters",
                content=(
                    "HNSW indexes accept two build parameters: m (default 16) and "
                    "ef_construction (default 64). At query time, hnsw.ef_search defaults to 40."
                ),
                final_score=0.88,
            ),
            RetrievedChunk(
                chunk_id="doc_pg::chunk_002",
                document_id="doc_pg",
                document_title="pgvector HNSW Indexing",
                source="pgvector.md",
                chunk_index=2,
                heading="Distance Operators",
                content=(
                    "pgvector provides three primary distance operators: <=> for cosine distance, "
                    "<-> for L2 Euclidean distance, and <#> for negative inner product."
                ),
                final_score=0.72,
            ),
        ]

    def test_extract_and_verify_citations_strips_out_of_range_tags(self) -> None:
        question = "What are the default m and ef_construction parameters for HNSW?"
        raw_answer = "HNSW uses m=16 and ef_construction=64 by default [1], alongside cosine distance [2] [99]."
        cleaned, citations = extract_and_verify_citations(question, raw_answer, self.chunks)
        self.assertNotIn("[99]", cleaned)
        self.assertIn("[1]", cleaned)
        self.assertIn("[2]", cleaned)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0].chunk_id, "doc_pg::chunk_001")
        self.assertEqual(citations[1].chunk_id, "doc_pg::chunk_002")

    def test_grounded_answer_generation_with_valid_evidence(self) -> None:
        question = "What are the default m and ef_construction parameters for HNSW?"
        answer, abstained, citations, _ = generate_grounded_answer(question, self.chunks)
        self.assertFalse(abstained)
        self.assertIn("[1]", answer)
        self.assertIn("16", answer)
        self.assertIn("64", answer)
        self.assertGreaterEqual(len(citations), 1)
        self.assertEqual(citations[0].chunk_id, "doc_pg::chunk_001")

    def test_abstention_on_unanswerable_question(self) -> None:
        question = "How do I configure BGP route reflectors in Cisco IOS XR?"
        answer, abstained, citations, _ = generate_grounded_answer(question, self.chunks)
        self.assertTrue(abstained)
        self.assertEqual(len(citations), 0)
        self.assertIn("cannot answer", answer.lower())


if __name__ == "__main__":
    unittest.main()
