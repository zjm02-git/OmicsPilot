"""阈值约束的单元测试（纯函数，不依赖数据库 / Redis / LLM）。

守三件事：
  1. 允许区间、预设档位、推荐区间三者**互相自洽**。改了常量必须同步，
     否则会出现「前端能选 / 后端拒绝」或者反过来——都是用户可见的坏事；
  2. 越界、非数字、布尔值一律拒绝。布尔尤其要守：True 是 int 的子类，
     不显式挡掉就会静默变成 1.0；
  3. None 表示「这一项沿用当前值」，必须放行——前端只改一项是常见路径。

用 unittest 而不是 pytest：项目 requirements.txt 里没有 pytest，
全仓测试都是 unittest 风格，`python -m unittest discover -s tests` 一把跑完。
"""

import math
import unittest

from omics_agent import thresholds


class TestBoundsConsistency(unittest.TestCase):
    """常量之间不能互相打脸。"""

    def test_defaults_inside_bounds(self):
        self.assertLessEqual(thresholds.FC_MIN, thresholds.FC_DEFAULT)
        self.assertLessEqual(thresholds.FC_DEFAULT, thresholds.FC_MAX)
        self.assertLessEqual(thresholds.FDR_MIN, thresholds.FDR_DEFAULT)
        self.assertLessEqual(thresholds.FDR_DEFAULT, thresholds.FDR_MAX)

    def test_recommended_inside_bounds(self):
        """推荐区间必须在允许区间内，否则页面上会出现「建议填一个填不进去的值」。"""
        fc_low, fc_high = thresholds.FC_RECOMMENDED
        fdr_low, fdr_high = thresholds.FDR_RECOMMENDED
        self.assertLessEqual(thresholds.FC_MIN, fc_low)
        self.assertLessEqual(fc_low, fc_high)
        self.assertLessEqual(fc_high, thresholds.FC_MAX)
        self.assertLessEqual(thresholds.FDR_MIN, fdr_low)
        self.assertLessEqual(fdr_low, fdr_high)
        self.assertLessEqual(fdr_high, thresholds.FDR_MAX)

    def test_presets_accepted(self):
        """HITL 的三档预设必须全部通过校验，否则用户点「标准」都会被拒。"""
        from omics_agent.nodes import THRESHOLD_PRESETS

        self.assertTrue(THRESHOLD_PRESETS, "预设档位不能为空，前端靠它渲染单选")
        for preset in THRESHOLD_PRESETS:
            with self.subTest(preset=preset.get("name")):
                check = thresholds.validate(
                    preset.get("fc_threshold"), preset.get("fdr_threshold")
                )
                self.assertTrue(check.ok, f"{preset.get('name')} 被拒：{check.message}")

    def test_bounds_payload_shape(self):
        """前端按这些键渲染输入框，字段缺一个就是 KeyError。"""
        payload = thresholds.bounds()
        for key in ("fc_threshold", "fdr_threshold"):
            self.assertIn(key, payload)
            item = payload[key]
            for field in ("label", "min", "max", "step", "default", "recommended", "help"):
                self.assertIn(field, item, f"{key} 缺字段 {field}")
            self.assertLessEqual(item["min"], item["default"])
            self.assertLessEqual(item["default"], item["max"])
            self.assertGreater(item["step"], 0)
            self.assertEqual(len(item["recommended"]), 2)
            self.assertTrue(item["help"])


class TestAcceptValid(unittest.TestCase):
    """合法值必须放过，包括两个端点。"""

    def test_default_and_common_values(self):
        for fc, fdr in [
            (thresholds.FC_DEFAULT, thresholds.FDR_DEFAULT),
            (0.585, 0.02),   # 文献里最常见的「1.5 倍 + FDR<0.02」
            (0.8, 0.05),
            (1.2, 0.01),
            (2.0, 0.001),
        ]:
            with self.subTest(fc=fc, fdr=fdr):
                self.assertTrue(thresholds.validate(fc, fdr).ok)

    def test_endpoints_accepted(self):
        """区间是闭区间：正好等于 min / max 必须通过。"""
        self.assertTrue(
            thresholds.validate(thresholds.FC_MIN, thresholds.FDR_MIN).ok
        )
        self.assertTrue(
            thresholds.validate(thresholds.FC_MAX, thresholds.FDR_MAX).ok
        )

    def test_int_value_accepted(self):
        """整数 1 也是合法输入（前端可能传 int 而不是 float）。"""
        self.assertTrue(thresholds.validate(1, 0.05).ok)


class TestRejectInvalid(unittest.TestCase):
    """越界与非数字必须拒绝，且给出中文原因。"""

    def test_fold_change_out_of_range(self):
        for bad in (-0.1, 5.1, 999):
            with self.subTest(bad=bad):
                check = thresholds.validate(bad, None)
                self.assertFalse(check.ok)
                self.assertIn("log2FC", check.message)

    def test_fdr_out_of_range(self):
        for bad in (0.0005, 0.6, 1.0, 0.0):
            with self.subTest(bad=bad):
                check = thresholds.validate(None, bad)
                self.assertFalse(check.ok)
                self.assertIn("FDR", check.message)

    def test_non_numeric_rejected(self):
        for bad in ("1.0", [1.0], {"v": 1}, ""):
            with self.subTest(bad=bad):
                self.assertFalse(thresholds.validate(bad, None).ok)
                self.assertFalse(thresholds.validate(None, bad).ok)

    def test_bool_is_not_a_number(self):
        """True 是 int 的子类，不显式挡掉会静默变成 1.0。"""
        self.assertFalse(thresholds.validate(True, None).ok)
        self.assertFalse(thresholds.validate(None, False).ok)

    def test_nan_and_inf_rejected(self):
        self.assertFalse(thresholds.validate(math.nan, None).ok)
        self.assertFalse(thresholds.validate(math.inf, None).ok)
        self.assertFalse(thresholds.validate(None, math.nan).ok)


class TestNoneMeansKeepCurrent(unittest.TestCase):
    """None = 「这一项沿用当前值」，不能报错。"""

    def test_both_none_is_ok(self):
        check = thresholds.validate(None, None)
        self.assertTrue(check.ok)
        self.assertEqual(check.message, "")

    def test_one_sided(self):
        self.assertTrue(thresholds.validate(0.8, None).ok)
        self.assertTrue(thresholds.validate(None, 0.02).ok)


class TestDescribe(unittest.TestCase):
    """报告里要能写清「按什么标准筛的」。"""

    def test_renders_human_text(self):
        text = thresholds.describe({"fc_threshold": 1.0, "fdr_threshold": 0.05})
        self.assertIn("log2FC", text)
        self.assertIn("FDR", text)
        self.assertIn("1", text)
        self.assertIn("0.05", text)

    def test_empty_input_falls_back_to_default(self):
        """传 None 时不能崩，要退回默认值渲染。"""
        text = thresholds.describe(None)
        self.assertIn("log2FC", text)
        self.assertIn("FDR", text)


if __name__ == "__main__":
    unittest.main()
