import unittest

from app.retrieval.hybrid import build_or_tsquery, reciprocal_rank_fusion
from app.schemas import RetrievedChunk


class TestHybridSearchAndRRF(unittest.TestCase):
    def test_build_or_tsquery_splits_identifiers_and_strips_stopwords(self) -> None:
        query = "What is the default ef_construction value for an HNSW index in pgvector?"
        tsquery = build_or_tsquery(query)
        self.assertIn("ef_construction", tsquery)
        self.assertIn("ef", tsquery)
        self.assertIn("construction", tsquery)
        self.assertIn("hnsw", tsquery)
        self.assertIn("pgvector", tsquery)
        self.assertNotIn("what", tsquery.split(" | "))
        self.assertNotIn("the", tsquery.split(" | "))

    def test_reciprocal_rank_fusion_boosts_overlapping_chunks(self) -> None:
        dense = [
            RetrievedChunk(
                chunk_id="doc1::chunk_000",
                document_id="doc1",
                document_title="Doc 1",
                source="doc1.md",
                chunk_index=0,
                content="First dense chunk",
                dense_rank=1,
                dense_score=0.91,
            ),
            RetrievedChunk(
                chunk_id="doc2::chunk_001",
                document_id="doc2",
                document_title="Doc 2",
                source="doc2.md",
                chunk_index=1,
                content="Consensus chunk present in both dense and sparse",
                dense_rank=2,
                dense_score=0.88,
            ),
        ]
        sparse = [
            RetrievedChunk(
                chunk_id="doc2::chunk_001",
                document_id="doc2",
                document_title="Doc 2",
                source="doc2.md",
                chunk_index=1,
                content="Consensus chunk present in both dense and sparse",
                sparse_rank=1,
                sparse_score=1.45,
            ),
            RetrievedChunk(
                chunk_id="doc3::chunk_000",
                document_id="doc3",
                document_title="Doc 3",
                source="doc3.md",
                chunk_index=0,
                content="Sparse only chunk",
                sparse_rank=2,
                sparse_score=0.75,
            ),
        ]

        fused = reciprocal_rank_fusion(dense, sparse, k=60, dense_weight=1.0, sparse_weight=1.0)
        self.assertEqual(len(fused), 3)
        # doc2::chunk_001 has rank 2 in dense (1/62) + rank 1 in sparse (1/61) = 0.032522
        # doc1::chunk_000 has rank 1 in dense (1/61) = 0.016393
        self.assertEqual(fused[0].chunk_id, "doc2::chunk_001")
        self.assertEqual(fused[0].dense_rank, 2)
        self.assertEqual(fused[0].sparse_rank, 1)
        self.assertAlmostEqual(fused[0].rrf_score, round(1 / 62 + 1 / 61, 6))
        self.assertEqual(fused[1].chunk_id, "doc1::chunk_000")
        self.assertEqual(fused[2].chunk_id, "doc3::chunk_000")


if __name__ == "__main__":
    unittest.main()
