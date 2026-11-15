import os
import unittest
from types import SimpleNamespace

from app.config import get_settings
from app.ingestion.embedder import _deterministic_feature_embedding
from app.ui_server import _is_admin_authorized
from eval.question_bank import _match_gold_ref_to_chunks
from eval.question_generator import (
    filter_answerability,
    filter_dedupe,
    filter_format,
    filter_lexical_overlap,
    filter_unanswerable,
    parse_generated_json,
)


class TestQuestionBankAndGenerator(unittest.TestCase):
    def test_parse_generated_json_variants_and_bad_inputs(self) -> None:
        self.assertEqual(parse_generated_json(""), [])
        self.assertEqual(parse_generated_json("not json at all"), [])
        self.assertEqual(parse_generated_json("{broken json"), [])

        raw_fence = 'Here you go:\n```json\n[{"question": "How fast does it charge?", "type": "paraphrase"}]\n```'
        parsed = parse_generated_json(raw_fence)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["question"], "How fast does it charge?")

        raw_obj = '{"questions": [{"question": "What is the warranty?", "type": "single_chunk"}]}'
        parsed_obj = parse_generated_json(raw_obj)
        self.assertEqual(len(parsed_obj), 1)
        self.assertEqual(parsed_obj[0]["type"], "single_chunk")

    def test_filter_format(self) -> None:
        self.assertFalse(filter_format("")[0])
        self.assertFalse(filter_format("No question mark")[0])
        self.assertFalse(filter_format("A" * 200 + "?")[0])
        self.assertTrue(filter_format("How do I clean the dishwasher filter?")[0])

    def test_filter_lexical_overlap_thresholds(self) -> None:
        src = ["Clean the filter at the bottom of the tub and check the drain hose for kinks."]
        # High overlap (~100% content words in source)
        copied = "Should I clean the filter at the bottom of the tub?"
        self.assertFalse(filter_lexical_overlap(copied, src, "paraphrase")[0])
        self.assertFalse(filter_lexical_overlap(copied, src, "single_chunk")[0])

        # ~75% overlap: rejected for paraphrase (>60%), allowed for single_chunk (<=80%)
        moderate = "Where is the filter inside the tub located?"
        # words: where(stop), filter(in), inside(out), tub(in), located(out) -> 2/4 = 0.50
        # Let's craft exact 3/4 = 0.75 overlap: filter(in), bottom(in), tub(in), repair(out)
        three_quarter = "How to repair the filter at the bottom of the tub?"
        self.assertFalse(filter_lexical_overlap(three_quarter, src, "paraphrase")[0])
        self.assertTrue(filter_lexical_overlap(three_quarter, src, "single_chunk")[0])

    def test_filter_answerability(self) -> None:
        src = ["F1: Water is not draining. Clean the filter at the bottom of the tub."]
        ok_pass, _ = filter_answerability(
            "Water is sitting at the bottom. What should I do?",
            ["clean the filter"],
            src,
            judge_fn=lambda _p: "You should clean the filter at the bottom of the tub.",
        )
        self.assertTrue(ok_pass)

        ok_abstain, _ = filter_answerability(
            "Water is sitting at the bottom. What should I do?",
            ["clean the filter"],
            src,
            judge_fn=lambda _p: "ABSTAIN",
        )
        self.assertFalse(ok_abstain)

        ok_miss, _ = filter_answerability(
            "Water is sitting at the bottom. What should I do?",
            ["clean the filter"],
            src,
            judge_fn=lambda _p: "Restart the machine.",
        )
        self.assertFalse(ok_miss)

    def test_filter_unanswerable(self) -> None:
        ok_unans, _ = filter_unanswerable(
            "Does the dishwasher have Bluetooth?",
            judge_fn=lambda _p: '{"answerable": false}',
            corpus_chunks_override=["Overview: 14 place settings, 44 dB."],
        )
        self.assertTrue(ok_unans)

        ok_ans, _ = filter_unanswerable(
            "How loud is the dishwasher?",
            judge_fn=lambda _p: '{"answerable": true}',
            corpus_chunks_override=["Overview: The noise level is 44 dB."],
        )
        self.assertFalse(ok_ans)

    def test_filter_dedupe(self) -> None:
        q1 = "How do I reset the Zephyr ZX-400 router?"
        v1 = _deterministic_feature_embedding(q1, 768)
        ok_dup, _, _ = filter_dedupe(q1, [v1], threshold=0.90)
        self.assertFalse(ok_dup)

        ok_new, _, _ = filter_dedupe("What is the noise level of the Aurora dishwasher?", [v1], threshold=0.90)
        self.assertTrue(ok_new)

    def test_reingest_gold_ref_resolution(self) -> None:
        ref = {
            "doc_id": "doc_test",
            "heading_path": "Guide > Display Codes",
            "quote_snippet": "Clean the filter at the bottom of the tub and check the drain hose for kinks.",
            "chunk_id": "doc_test::chunk_001",
        }
        new_chunks = [
            SimpleNamespace(id="doc_test::chunk_000", heading="Guide > Overview", content="14 place settings."),
            SimpleNamespace(
                id="doc_test::chunk_004",
                heading="Guide > Display Codes",
                content="F1: Clean the filter at the bottom of the tub and check the drain hose for kinks.",
            ),
        ]
        self.assertEqual(_match_gold_ref_to_chunks(ref, new_chunks), "doc_test::chunk_004")

        unrelated = [SimpleNamespace(id="doc_test::chunk_000", heading="Guide > Warranty", content="2 years warranty.")]
        self.assertIsNone(_match_gold_ref_to_chunks(ref, unrelated))

    def test_demo_mode_admin_auth(self) -> None:
        os.environ["DEMO_MODE"] = "true"
        os.environ["ADMIN_TOKEN"] = "secret-123"
        get_settings.cache_clear()
        try:
            self.assertFalse(_is_admin_authorized({}))
            self.assertFalse(_is_admin_authorized({"X-Admin-Token": "wrong"}))
            self.assertTrue(_is_admin_authorized({"X-Admin-Token": "secret-123"}))
            self.assertTrue(_is_admin_authorized({"Authorization": "Bearer secret-123"}))
        finally:
            os.environ.pop("DEMO_MODE", None)
            os.environ.pop("ADMIN_TOKEN", None)
            get_settings.cache_clear()


if __name__ == "__main__":
    unittest.main()
