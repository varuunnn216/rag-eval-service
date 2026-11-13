import unittest

from app.retrieval.reranker import rerank_chunks
from app.schemas import RetrievedChunk


class TestCrossEncoderReranker(unittest.TestCase):
    def test_reranker_promotes_exact_answering_chunk(self) -> None:
        query = "What is the default ef_construction value for an HNSW index in pgvector?"
        candidates = [
            RetrievedChunk(
                chunk_id="doc_pg::chunk_000",
                document_id="doc_pg",
                document_title="PostgreSQL Overview",
                source="pg.md",
                chunk_index=0,
                heading="General Indexing",
                content="PostgreSQL indexes improve query speed. Users can create B-tree or GiST indexes on tables.",
                dense_rank=1,
                rrf_score=0.031,
                final_score=0.031,
            ),
            RetrievedChunk(
                chunk_id="doc_pg::chunk_004",
                document_id="doc_pg",
                document_title="PostgreSQL Overview",
                source="pg.md",
                chunk_index=4,
                heading="Vector Indexes > HNSW Build Parameters",
                content=(
                    "When creating an HNSW index in pgvector, the default ef_construction value is 64 "
                    "and the default m parameter is 16."
                ),
                dense_rank=3,
                sparse_rank=2,
                rrf_score=0.029,
                final_score=0.029,
            ),
        ]

        reranked = rerank_chunks(query, candidates, top_k=2)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].chunk_id, "doc_pg::chunk_004")
        self.assertIsNotNone(reranked[0].rerank_score)
        self.assertGreater(reranked[0].rerank_score or 0.0, reranked[1].rerank_score or 0.0)
        # Ensure original retrieval metadata is preserved
        self.assertEqual(reranked[0].dense_rank, 3)
        self.assertEqual(reranked[0].sparse_rank, 2)


if __name__ == "__main__":
    unittest.main()
