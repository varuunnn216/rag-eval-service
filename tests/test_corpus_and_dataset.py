import json
import unittest

from app.config import get_settings
from app.ingestion.seed import load_corpus_documents, preview_corpus_chunks


class TestCorpusAndEvalDataset(unittest.TestCase):
    def test_corpus_loads_and_chunks_deterministically(self) -> None:
        docs = load_corpus_documents()
        self.assertEqual(len(docs), 8)
        chunks = preview_corpus_chunks()
        self.assertEqual(len(chunks), 35)

    def test_eval_dataset_has_40_questions_and_valid_ground_truth_chunk_ids(self) -> None:
        settings = get_settings()
        dataset_path = settings.project_root / "eval" / "dataset.json"
        self.assertTrue(dataset_path.exists())

        items = json.loads(dataset_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(items), 40)
        self.assertLessEqual(len(items), 65)
        self.assertEqual(len(items), 60)

        valid_chunk_ids = {c.id for c in preview_corpus_chunks()}
        valid_doc_ids = {d.id for d in load_corpus_documents()}

        categories = set()
        for item in items:
            categories.add(item["category"])
            if item["should_abstain"]:
                self.assertEqual(item["expected_chunk_ids"], [])
            else:
                self.assertIn(item["expected_document_id"], valid_doc_ids)
                self.assertGreaterEqual(len(item["expected_chunk_ids"]), 1)
                for cid in item["expected_chunk_ids"]:
                    self.assertIn(
                        cid,
                        valid_chunk_ids,
                        f"Question {item['id']} references unknown chunk_id {cid}",
                    )

        self.assertEqual(
            categories,
            {"exact_lookup", "semantic_conceptual", "multi_hop", "unanswerable"},
        )


if __name__ == "__main__":
    unittest.main()
