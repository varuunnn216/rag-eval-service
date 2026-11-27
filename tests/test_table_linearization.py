"""Unit tests for Markdown table linearization and contextual chunk formatting."""

import unittest

from app.ingestion.chunker import format_contextual_chunk_text, linearize_tables


class TestTableLinearization(unittest.TestCase):
    def test_passthrough_without_pipes(self) -> None:
        text = "Plain paragraph without any Markdown table."
        self.assertEqual(linearize_tables(text), text)

    def test_linearizes_markdown_table_and_handles_missing_cells(self) -> None:
        md = (
            "| Metric | Dense | Hybrid |\n"
            "| --- | --- | --- |\n"
            "| Recall@5 | 94.9% | 100.0% |\n"
            "| MRR@5 | 0.864 | |\n"
        )
        out = linearize_tables(md)
        self.assertIn("Columns: Metric, Dense, Hybrid.", out)
        self.assertIn("Metric: Recall@5. Dense: 94.9%. Hybrid: 100.0%.", out)
        self.assertIn("Metric: MRR@5. Dense: 0.864.", out)

    def test_format_contextual_chunk_text_deduplicates_matching_title(self) -> None:
        formatted = format_contextual_chunk_text(
            document_title="pgvector Guide",
            heading="pgvector Guide > HNSW",
            content="Default m is 16.",
            linearize=False,
        )
        self.assertEqual(formatted, "pgvector Guide > HNSW\nDefault m is 16.")

        prefixed = format_contextual_chunk_text(
            document_title="pgvector Guide",
            heading="HNSW",
            content="Default m is 16.",
            linearize=False,
        )
        self.assertEqual(prefixed, "pgvector Guide > HNSW\nDefault m is 16.")


if __name__ == "__main__":
    unittest.main()
