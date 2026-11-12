"""上下文预算层测试：token 估算、截断、按优先级裁剪。"""

import unittest

from omics_agent import context


class EstimateTokensTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(context.estimate_tokens(""), 0)

    def test_cjk_is_heavier_than_ascii(self):
        cn = context.estimate_tokens("差异分析报告")
        en = context.estimate_tokens("abcde")
        self.assertGreater(cn, en)

    def test_monotonic(self):
        short = context.estimate_tokens("基因")
        long = context.estimate_tokens("基因" * 50)
        self.assertGreater(long, short)


class TruncateTokensTest(unittest.TestCase):
    def test_short_text_untouched(self):
        text = "短文本"
        self.assertEqual(context.truncate_tokens(text, 100), text)

    def test_zero_budget(self):
        self.assertEqual(context.truncate_tokens("abc", 0), "")

    def test_truncated_within_budget(self):
        text = "基因表达差异分析" * 200
        out = context.truncate_tokens(text, 120)
        self.assertLessEqual(context.estimate_tokens(out), 120)
        self.assertLess(len(out), len(text))
        self.assertTrue(out.endswith("…"))


class FitBlocksTest(unittest.TestCase):
    def _blocks(self):
        return [
            {"name": "diff", "text": "差" * 200, "priority": 3},
            {"name": "enrich", "text": "富" * 200, "priority": 2},
            {"name": "genes", "text": "基" * 200, "priority": 1},
        ]

    def test_all_fit_when_budget_is_large(self):
        texts, info = context.fit_blocks(self._blocks(), 10_000)
        self.assertEqual(info["dropped"], [])
        self.assertEqual(info["truncated"], [])
        self.assertTrue(all(texts.values()))
        self.assertLessEqual(info["used_tokens"], 10_000)

    def test_drops_lowest_priority_first(self):
        texts, info = context.fit_blocks(self._blocks(), 450)
        self.assertIn("genes", info["dropped"])
        self.assertTrue(texts["diff"])
        self.assertLessEqual(info["used_tokens"], 450)

    def test_never_drops_the_only_block_but_truncates(self):
        blocks = [{"name": "diff", "text": "差" * 800, "priority": 3}]
        texts, info = context.fit_blocks(blocks, 100)
        self.assertEqual(info["dropped"], [])
        self.assertIn("diff", info["truncated"])
        self.assertTrue(texts["diff"])
        self.assertLessEqual(info["used_tokens"], 100)

    def test_empty_block_is_ignored(self):
        texts, info = context.fit_blocks(
            [{"name": "empty", "text": "", "priority": 9},
             {"name": "diff", "text": "差" * 10, "priority": 3}], 100)
        self.assertEqual(info["dropped"], [])
        self.assertEqual(texts["empty"], "")


class DropNoteTest(unittest.TestCase):
    def test_empty_when_nothing_dropped(self):
        self.assertEqual(context.drop_note({"dropped": [], "truncated": []}), "")

    def test_mentions_names(self):
        note = context.drop_note({"dropped": ["genes"], "truncated": ["enrich"]})
        self.assertIn("genes", note)
        self.assertIn("enrich", note)


if __name__ == "__main__":
    unittest.main()
