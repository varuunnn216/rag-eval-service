import unittest

from app.schemas import Citation, RetrievedChunk
from eval.metrics import (
    citation_metrics,
    fact_coverage_score,
    faithfulness_score,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class TestEvalMetrics(unittest.TestCase):
    def test_retrieval_metrics_recall_mrr_ndcg(self) -> None:
        retrieved = ["c2", "c1", "c3", "c4", "c5"]
        expected = ["c1", "c4"]

        self.assertEqual(recall_at_k(retrieved, expected, 1), 0.0)
        self.assertEqual(recall_at_k(retrieved, expected, 2), 0.5)
        self.assertEqual(recall_at_k(retrieved, expected, 5), 1.0)
        self.assertEqual(hit_rate_at_k(retrieved, expected, 1), 0.0)
        self.assertEqual(hit_rate_at_k(retrieved, expected, 2), 1.0)
        self.assertEqual(mrr_at_k(retrieved, expected, 5), 0.5)
        self.assertGreater(ndcg_at_k(retrieved, expected, 5), 0.6)

    def test_fact_coverage_and_citation_metrics(self) -> None:
        ans = "HNSW uses m = 16 and ef_construction = 64 by default [1]."
        score = fact_coverage_score(ans, ["16", "64", "ef_construction"], should_abstain=False, abstained=False)
        self.assertEqual(score, 1.0)

        citations = [
            Citation(
                citation_number=1,
                chunk_id="doc_pg::chunk_002",
                document_id="doc_pg",
                document_title="pgvector",
                source="pg.md",
                quote="m = 16 and ef_construction = 64",
            )
        ]
        cit_m = citation_metrics(citations, ["doc_pg::chunk_002"], should_abstain=False, abstained=False)
        self.assertEqual(cit_m["citation_precision"], 1.0)
        self.assertEqual(cit_m["citation_recall"], 1.0)

        chunks = [
            RetrievedChunk(
                chunk_id="doc_pg::chunk_002",
                document_id="doc_pg",
                document_title="pgvector",
                source="pg.md",
                chunk_index=2,
                heading="HNSW",
                content="HNSW uses m = 16 and ef_construction = 64 by default.",
            )
        ]
        faith = faithfulness_score(ans, citations, chunks, abstained=False)
        self.assertEqual(faith, 1.0)


if __name__ == "__main__":
    unittest.main()
