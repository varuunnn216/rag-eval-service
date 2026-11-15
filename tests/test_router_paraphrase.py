import os
import unittest

from app.generation.generator import ABSTENTION_MESSAGE, answer_question, set_custom_llm_caller
from app.ingestion.chunker import (
    chunk_document,
    format_contextual_chunk_text,
    linearize_tables,
)
from app.retrieval.query_expansion import expand_query
from app.schemas import AskRequest


class TestParaphraseAndTableFixes(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_disable = os.environ.get("DISABLE_NEURAL_RERANKER")
        os.environ["DISABLE_NEURAL_RERANKER"] = "1"

    def tearDown(self) -> None:
        set_custom_llm_caller(None)
        if self._prev_disable is None:
            os.environ.pop("DISABLE_NEURAL_RERANKER", None)
        else:
            os.environ["DISABLE_NEURAL_RERANKER"] = self._prev_disable

    def test_linearize_tables_handles_ragged_rows_without_crash(self) -> None:
        table = (
            "| LED            | Meaning                              | What to do                       |\n"
            "|----------------|--------------------------------------|----------------------------------|\n"
            "| Blinking red   | Hardware fault                       | Contact support                  |\n"
            "| Solid blue     | WPS pairing is active for 2 minutes |\n"
        )
        out = linearize_tables(table)
        self.assertIn("LED: Blinking red. Meaning: Hardware fault. What to do: Contact support.", out)
        self.assertIn("LED: Solid blue. Meaning: WPS pairing is active for 2 minutes.", out)

    def test_chunk_document_never_splits_table(self) -> None:
        doc = (
            "# Guide\n\n"
            "## LED Indicators\n\n"
            "| LED | Meaning | What to do |\n"
            "|---|---|---|\n"
            "| Solid green | Online and working | Nothing |\n"
            "| Blinking green | Starting up | Wait up to 2 minutes |\n"
            "| Solid red | No internet connection | Check the modem cable, then restart the router |\n"
            "| Blinking red | Hardware fault | Contact support |\n"
            "| Solid blue | WPS pairing is active for 2 minutes |\n"
            "| Off | No power | Check the power adapter |\n"
        )
        chunks = chunk_document("doc_test_table", doc, max_tokens=20, overlap_tokens=0, min_chunk_tokens=0)
        self.assertEqual(len(chunks), 1)
        self.assertIn("| Blinking red   | Hardware fault" if False else "| Blinking red | Hardware fault", chunks[0].content)

    def test_contextual_chunk_header_formatting(self) -> None:
        ctx = format_contextual_chunk_text(
            "Zephyr ZX-400 Router Guide",
            "LED Indicators",
            "| LED | Meaning |\n|---|---|\n| Blinking red | Hardware fault |",
            linearize=True,
        )
        self.assertTrue(ctx.startswith("Zephyr ZX-400 Router Guide > LED Indicators\n"))
        self.assertIn("LED: Blinking red. Meaning: Hardware fault.", ctx)

    def test_query_expansion_generates_synonym_rewrites(self) -> None:
        rewrites = expand_query("The light on my router keeps flashing red. What should I do?")
        joined = " ".join(rewrites).lower()
        self.assertIn("blinking", joined)
        self.assertIn("led", joined)

    def test_paraphrase_questions_and_traps_with_adaptive_fallback(self) -> None:
        import re
        from typing import Optional

        def find_block_num(prompt: str, needle: str) -> Optional[str]:
            for block in prompt.split("\n---\n"):
                if needle in block:
                    m = re.search(r"\[(\d+)\]", block)
                    if m:
                        return m.group(1)
            return None

        def mock_llm(prompt: str, system: str):
            self.assertIn("Treat synonyms and different wording as equivalent", system)
            if "flashing red" in prompt:
                n = find_block_num(prompt, "LED: Blinking red. Meaning: Hardware fault. What to do: Contact support.")
                if n:
                    return f"Blinking red indicates a hardware fault, so you should contact support [{n}].", 50, 15
            if "wipe" in prompt:
                n = find_block_num(prompt, "hold the Reset button for 10 seconds")
                if n:
                    return f"Hold the Reset button for 10 seconds while powered on [{n}].", 50, 15
            if "shutting itself off" in prompt:
                n = find_block_num(prompt, "E-407: The router is overheating")
                if n:
                    return f"The router is overheating (E-407) and shuts down at 70 degrees Celsius [{n}].", 50, 15
            if "money back" in prompt:
                n = find_block_num(prompt, "opened router can be returned within 15 days")
                if n:
                    return f"An opened router can be returned within 15 days with a 10 percent restocking fee [{n}].", 50, 15
            if "collect an item" in prompt:
                n = find_block_num(prompt, "Collect the item within 5 days")
                if n:
                    return f"You have 5 days from the arrival email to collect the item [{n}].", 50, 15
            return ABSTENTION_MESSAGE, 40, 10

        set_custom_llm_caller(mock_llm)

        paraphrases = [
            ("The light on my router keeps flashing red. What should I do?", "hardware fault"),
            ("How do I wipe all my settings on the router?", "10 seconds"),
            ("My router keeps shutting itself off. Why?", "E-407"),
            ("Can I get my money back on a router I already opened?", "15 days"),
            ("How many days do I have to collect an item I requested from another library?", "5 days"),
        ]
        for q, expected_substr in paraphrases:
            resp = answer_question(AskRequest(question=q, mode="hybrid", use_reranker=True, query_expansion=True))
            self.assertFalse(resp.abstained, f"Unexpected abstention on: {q}")
            self.assertIn(expected_substr, resp.answer)
            self.assertGreaterEqual(resp.verified_citations_count, 1)

        for trap_q in [
            "Does the ZX-400 support Wi-Fi 7?",
            "What CPU does the router use?",
        ]:
            resp = answer_question(AskRequest(question=trap_q, mode="hybrid", use_reranker=True, query_expansion=True))
            self.assertTrue(resp.abstained, f"Trap question should abstain: {trap_q}")
            self.assertEqual(resp.citations, [])
            self.assertIsNotNone(resp.abstain_reason)


if __name__ == "__main__":
    unittest.main()
