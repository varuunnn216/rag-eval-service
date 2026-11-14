import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

JSC_PATH = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc"
APP_JS_PATH = Path(__file__).resolve().parent.parent / "app" / "static" / "app.js"


def _run_jsc_snippet(js_code: str):
    """Execute renderSafeMarkdown & helpers from app.js inside macOS JavaScriptCore."""
    src = APP_JS_PATH.read_text(encoding="utf-8")
    start = src.index("function escapeHtml(")
    end = src.index("/* =====================================================================\n   TABS, SLIDING NAV PILL")
    helpers_js = src[start:end]

    script = f"""
var console = {{ error: function() {{}}, log: function() {{}} }};
{helpers_js}
{js_code}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    try:
        proc = subprocess.run([JSC_PATH, tmp_path], capture_output=True, text=True, check=True)
        return json.loads(proc.stdout.strip())
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@unittest.skipUnless(os.path.exists(JSC_PATH), "JavaScriptCore jsc CLI not available")
class TestMarkdownRenderer(unittest.TestCase):
    def test_single_orphan_table_row_does_not_throw(self):
        out = _run_jsc_snippet(
            'print(JSON.stringify(renderSafeMarkdown("| Graduate      | 30            | 60 days     | 3         |")));'
        )
        self.assertIn("<table", out)
        self.assertIn("<td>Graduate</td>", out)
        self.assertIn("<td>60 days</td>", out)

    def test_ragged_table_rows_fewer_and_extra_cells(self):
        md = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 |\n"
            "| 1 | 2 | 3 | 4 |"
        )
        out = _run_jsc_snippet(f"print(JSON.stringify(renderSafeMarkdown({json.dumps(md)})));")
        self.assertIn("<thead>", out)
        self.assertEqual(out.count("<th>"), 4)
        self.assertEqual(out.count("<td>"), 8)

    def test_table_cut_mid_row_without_trailing_pipe(self):
        md = (
            "| Borrower | Books |\n"
            "|---|---|\n"
            "| Graduate | 30"
        )
        out = _run_jsc_snippet(f"print(JSON.stringify(renderSafeMarkdown({json.dumps(md)})));")
        self.assertIn("<table", out)
        self.assertIn("<td>Graduate</td>", out)
        self.assertIn("<td>30</td>", out)

    def test_empty_table(self):
        md = "| Col 1 | Col 2 |\n|---|---|"
        out = _run_jsc_snippet(f"print(JSON.stringify(renderSafeMarkdown({json.dumps(md)})));")
        self.assertIn("<thead>", out)
        self.assertIn("<tbody></tbody>", out)

    def test_nested_lists(self):
        md = (
            "- Parent 1\n"
            "  - Child A\n"
            "    - Grandchild\n"
            "- Parent 2"
        )
        out = _run_jsc_snippet(f"print(JSON.stringify(renderSafeMarkdown({json.dumps(md)})));")
        self.assertEqual(out.count("<ul"), 3)
        self.assertIn("Grandchild", out)

    def test_unclosed_code_fence_cut_off(self):
        md = "Before\n```python\ndef hello():\n  return 42"
        out = _run_jsc_snippet(f"print(JSON.stringify(renderSafeMarkdown({json.dumps(md)})));")
        self.assertIn("md-code-block", out)
        self.assertIn("return 42", out)

    def test_clamp_array_len_and_chunk_preview_boundary(self):
        res = _run_jsc_snippet(
            """
            var clamped = [clampArrayLen(-5), clampArrayLen(NaN), clampArrayLen(undefined), clampArrayLen(3.9)];
            var tableMd = "| A | B |\\n|---|---|\\n| row1 | val1 |\\n| row2 | val2 |\\n| row3 | val3 |\\n| row4 | val4 |";
            var preview = truncateChunkPreview(tableMd, 45);
            print(JSON.stringify({ clamped: clamped, preview: preview }));
            """
        )
        self.assertEqual(res["clamped"], [0, 0, 0, 3])
        self.assertTrue(res["preview"]["truncated"])
        # Must end on a complete table row
        self.assertTrue(res["preview"]["previewText"].endswith("|"))
