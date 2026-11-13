import unittest

from app.ingestion.chunker import chunk_document, estimate_token_count
from app.ingestion.embedder import _deterministic_feature_embedding


class TestChunkerAndEmbedder(unittest.TestCase):
    def test_heading_hierarchy_and_deterministic_ids(self) -> None:
        sample_md = """# PostgreSQL Indexing Guide

PostgreSQL supports several index types including B-tree, Hash, GiST, SP-GiST, GIN, and BRIN.

## Vector Indexes (pgvector)

### HNSW Parameters
HNSW builds a multilayer graph using `m` (max links per node, default 16) and `ef_construction` (candidate list size during build, default 64).
"""
        chunks = chunk_document("doc_pg_index", sample_md, max_tokens=120, overlap_tokens=20)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].id, "doc_pg_index::chunk_000")
        self.assertEqual(chunks[0].heading, "PostgreSQL Indexing Guide")
        self.assertEqual(chunks[1].id, "doc_pg_index::chunk_001")
        self.assertEqual(
            chunks[1].heading,
            "PostgreSQL Indexing Guide > Vector Indexes (pgvector) > HNSW Parameters",
        )
        self.assertIn("ef_construction", chunks[1].content)

    def test_token_window_splitting_and_overlap(self) -> None:
        para_1 = " ".join(["alpha"] * 40)
        para_2 = " ".join(["beta"] * 40)
        para_3 = " ".join(["gamma"] * 40)
        text = f"# Section One\n\n{para_1}\n\n{para_2}\n\n{para_3}"

        chunks = chunk_document("doc_overlap", text, max_tokens=120, overlap_tokens=60)
        self.assertGreaterEqual(len(chunks), 2)
        # Verify para_2 is carried over into the second chunk via overlap
        self.assertIn("beta", chunks[0].content)
        self.assertIn("beta", chunks[1].content)
        self.assertIn("gamma", chunks[1].content)
        self.assertGreater(estimate_token_count(chunks[0].content), 0)

    def test_deterministic_embedding_similarity(self) -> None:
        v1 = _deterministic_feature_embedding("HNSW index ef_construction parameter in pgvector", 768)
        v2 = _deterministic_feature_embedding("Configuring pgvector HNSW ef_construction parameter", 768)
        v3 = _deterministic_feature_embedding("OAuth2 JWT bearer token expiration policy", 768)

        sim_related = sum(a * b for a, b in zip(v1, v2))
        sim_unrelated = sum(a * b for a, b in zip(v1, v3))
        self.assertEqual(len(v1), 768)
        self.assertGreater(sim_related, 0.45)
        self.assertGreater(sim_related, sim_unrelated)


if __name__ == "__main__":
    unittest.main()
