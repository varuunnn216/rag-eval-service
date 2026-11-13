import unittest

from app.generation.generator import (
    ABSTENTION_MESSAGE,
    select_context_chunks,
    verify_answer_citations,
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
                    "HNSW indexes construct a multi-layer proximity graph. "
                    "HNSW indexes accept two build parameters: m (default 16) and "
                    "ef_construction (default 64). At query time, hnsw.ef_search defaults to 40."
                ),
                rerank_score=0.88,
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
                rerank_score=0.72,
                final_score=0.72,
            ),
        ]

    def test_verify_answer_citations_selects_supporting_sentence_and_drops_unselected(self) -> None:
        raw_answer = (
            "HNSW uses m=16 and ef_construction=64 by default [1], "
            "alongside cosine distance [2] [99]."
        )
        cleaned, citations, verified_cnt, total_cnt = verify_answer_citations(
            raw_answer, self.chunks
        )
        self.assertNotIn("[99]", cleaned)
        self.assertIn("[1]", cleaned)
        self.assertIn("[2]", cleaned)
        self.assertEqual(verified_cnt, 2)
        self.assertEqual(total_cnt, 3)  # [1], [2], and dropped [99]
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0].chunk_id, "doc_pg::chunk_001")
        # Supporting sentence must be the second sentence with m/ef_construction, not the first sentence
        self.assertIn("ef_construction", citations[0].quote)
        self.assertTrue(citations[0].verified)

    def test_context_selection_abstains_when_top_score_below_abstain_floor(self) -> None:
        low_chunks = [
            RetrievedChunk(
                chunk_id="doc_pg::chunk_001",
                document_id="doc_pg",
                document_title="pgvector",
                source="pg.md",
                chunk_index=1,
                content="Unrelated chunk",
                rerank_score=0.05,
                final_score=0.05,
            )
        ]
        annotated, selected, abstain_pre_llm = select_context_chunks(low_chunks)
        self.assertTrue(abstain_pre_llm)
        self.assertEqual(len(selected), 0)
        self.assertFalse(annotated[0].selected)
        self.assertEqual(annotated[0].selection_status, "dropped")
        self.assertEqual(ABSTENTION_MESSAGE, "I don't have enough information in the provided documents.")


if __name__ == "__main__":
    unittest.main()
