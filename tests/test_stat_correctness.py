"""统计正确性的金标测试。

守住三件曾经真错过、而且是"错得看不出来"的事：

  1. log2FC 的值空间：表达矩阵是 log2(FPKM+1) 时只能相减，
     再套一层 log2(比值) 就是双重取对数——数字看着合理，但没有倍数语义。
  2. log2FC 的方向：正数 = 在 group_b 中更高。报告必须写出参照组，
     否则读者会把"在 normal 中更高"读成"在 tumor 中上调"。
  3. 富集里 BH 的检验数 m 不能被 k==0 的通路提前裁掉，
     否则所有报出来的 FDR 都系统性偏小（anti-conservative）。
"""

import unittest

import numpy as np
import pandas as pd

from omics_agent.diff import differential_expression
from omics_agent.enrichment import enrichment_analysis

LOG2_3 = float(np.log2(3.0))  # 3 倍变化在 log2 空间的差值 ≈ 1.585


def _expr(scale, n_genes=40, n_per_group=40, fold=LOG2_3, seed=11):
    """造一份表达矩阵 + 分组表；G000 在 B 组比 A 组高 fold（log2 单位）。"""
    rng = np.random.default_rng(seed)
    samples = [f"S{i:02d}" for i in range(n_per_group * 2)]
    genes = [f"G{i:03d}" for i in range(n_genes)]

    if scale == "log2":
        mat = rng.normal(8.0, 0.15, size=(n_genes, len(samples)))
        mat[0, n_per_group:] += fold
    else:
        mat = rng.normal(100.0, 5.0, size=(n_genes, len(samples)))
        mat[0, n_per_group:] *= 2.0 ** fold

    expr = pd.DataFrame(mat, columns=samples)
    expr.insert(0, "gene", genes)
    meta = pd.DataFrame(
        {
            "sample_id": samples,
            "condition": ["A"] * n_per_group + ["B"] * n_per_group,
        }
    )
    return expr, meta


class Log2FoldChangeTest(unittest.TestCase):
    def test_log2_scale_is_difference_not_ratio(self):
        """log2 量纲下 log2FC 必须是均值之差，不能是 log2(均值之比)。"""
        expr, meta = _expr("log2")
        res = differential_expression(expr, meta, "condition", "A", "B")
        fc = float(res[res["gene"] == "G000"].iloc[0]["log2fc"])
        # 容差按采样噪声给（每组 40 个样本、噪声 sd=0.15 → 差值 se≈0.034），
        # 而双重取对数的旧口径会给出 ~0.26，两者差一个数量级，容差拦得住。
        self.assertAlmostEqual(fc, LOG2_3, delta=0.15,
                               msg="log2FC 不等于均值之差，疑似又对 log2 数据取了 log2(比值)")

    def test_direction_is_b_relative_to_a(self):
        """正数 = 在 group_b 中更高；交换两组必须变号。"""
        expr, meta = _expr("log2")
        ab = differential_expression(expr, meta, "condition", "A", "B")
        ba = differential_expression(expr, meta, "condition", "B", "A")
        fwd = float(ab[ab["gene"] == "G000"].iloc[0]["log2fc"])
        rev = float(ba[ba["gene"] == "G000"].iloc[0]["log2fc"])
        self.assertGreater(fwd, 1.0)
        self.assertLess(rev, -1.0)
        self.assertAlmostEqual(fwd, -rev, places=2)

    def test_linear_scale_uses_ratio(self):
        """线性量纲才走 log2(比值)，3 倍变化仍应得到 ≈1.585。"""
        expr, meta = _expr("linear")
        res = differential_expression(
            expr, meta, "condition", "A", "B", input_scale="linear"
        )
        fc = float(res[res["gene"] == "G000"].iloc[0]["log2fc"])
        self.assertGreater(fc, 1.5)
        self.assertLess(fc, 1.62)


class EnrichmentBHTest(unittest.TestCase):
    def _fixture(self):
        bg = [f"BG{i:03d}" for i in range(100)]
        sig = [f"SG{i:03d}" for i in range(10)]
        big = [f"X{i:03d}" for i in range(600)]
        background = bg + sig + big

        rows = []
        # K=30，一个显著基因都没有 → k==0，p=1，但必须留在 m 里
        for g in bg[:30]:
            rows.append({"gene": g, "term": "T_K0"})
        # K=10，其中 5 个显著
        for g in sig[:5] + bg[30:35]:
            rows.append({"gene": g, "term": "T_HIT"})
        # K=2，太小，应被过滤
        for g in sig[:2]:
            rows.append({"gene": g, "term": "T_TINY"})
        # K=600，太大（泛类目），应被过滤
        for g in big:
            rows.append({"gene": g, "term": "T_BIG"})

        return sig[:5], pd.DataFrame(rows), background

    def test_k0_terms_stay_in_denominator(self):
        """k==0 的通路要留在结果里（p=1），否则 BH 的 m 被低估。"""
        sig, ann, background = self._fixture()
        res = enrichment_analysis(sig, ann, background)
        terms = set(res["term"])
        self.assertIn("T_K0", terms, "k==0 的通路被提前裁掉，BH 的 m 会偏小、FDR 偏乐观")
        k0 = res[res["term"] == "T_K0"].iloc[0]
        self.assertAlmostEqual(float(k0["pvalue"]), 1.0, places=9)
        self.assertEqual(int(k0["sig_count"]), 0)

    def test_term_size_bounds(self):
        """基因集大小要有上下限，否则 top 榜会被噪声和泛类目占满。"""
        sig, ann, background = self._fixture()
        res = enrichment_analysis(sig, ann, background)
        terms = set(res["term"])
        self.assertNotIn("T_TINY", terms, "K<5 的通路没有被过滤")
        self.assertNotIn("T_BIG", terms, "K>500 的通路没有被过滤")
        self.assertIn("T_HIT", terms)
        hit = res[res["term"] == "T_HIT"].iloc[0]
        self.assertEqual(int(hit["sig_count"]), 5)
        self.assertEqual(int(hit["bg_count"]), 10)

    def test_denominator_size_is_observable(self):
        """只有 T_HIT 与 T_K0 通过过滤 → m 必须是 2。"""
        sig, ann, background = self._fixture()
        res = enrichment_analysis(sig, ann, background)
        self.assertEqual(len(res), 2, "参与 BH 的通路数不对，m 口径可能又被改了")


if __name__ == "__main__":
    unittest.main()
