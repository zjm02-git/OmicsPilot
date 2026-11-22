"""差异表达分析：纯 Python 统计计算（不依赖 LLM）。

核心思想：对每个基因，比较 A 组和 B 组的表达量，判断它是不是"显著差异"。
四个产出：
  1. log2FC（fold change，倍数变化）—— 表达量差了多少倍，取 log2 为了对称；
     现在带 95% 置信区间（CI）：区间跨过 0，说明连"往哪边变"都还不确定，
     比只看"显著/不显著"更能看出结论有多稳；
  2. p 值（Welch t 检验）—— 差异是不是"碰巧"出现的概率；
  3. FDR（BH 校正后的 p 值）—— 一次比较成千上万个基因，
     必须做多重检验校正，否则假阳性一大堆；
  4. 独立过滤（independent filtering）—— 检验前先剔除两组都低表达的基因，
     被剔除的基因**不进入 BH 的检验数 m**。这是 limma 的 filterByExpr、
     DESeq2 的 independentFiltering 在做的事：上万個本来就几乎没表达的基因
     既测不出稳定差异，又只会把 m 抬高，让真正有差异的基因丢掉显著性。

关于缺失值（NaN）：表达矩阵里的 NaN 表示"这个样本没测到这个基因"
（fetch_tcga.py 现在把这种格子留空，旧版是补成 0）。所有统计一律走 nan 语义
（nanmean / nanvar / 逐基因的可用样本数），绝不把 NaN 当 0 参与均值：
把 NaN 当 0 会让该基因在缺样本那一组的均值被拉低，凭空造出上调或下调。

关于显著性检验：优先用 scipy 的 Welch t 检验（更专业）；
如果没装 scipy，自动退化为正态近似（z 检验），保证项目开箱即跑。
"""

import math
import warnings
from statistics import NormalDist
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_NORMAL = NormalDist()  # 没有 scipy 时用来取 CI 的正态分位点（95% → 1.96）


class _Welch(NamedTuple):
    """一个基因两组比较的全部中间量（都是按基因排列的数组）。"""

    n_a: np.ndarray
    n_b: np.ndarray
    mean_a: np.ndarray
    mean_b: np.ndarray
    var_a: np.ndarray
    var_b: np.ndarray
    se: np.ndarray
    df: np.ndarray
    pvalue: np.ndarray


def _group_stats(mat: np.ndarray):
    """逐基因算 (可用样本数, 均值, 方差)，NaN 一律跳过。

    可用样本数必须逐基因算：缺失是按格子发生的，不同基因能用的样本数不同。
    如果仍拿全局样本数当分母，方差会被系统性低估，p 值随之偏小——
    这和"把 NaN 当 0"是同一类错误的两种表现。
    """
    n = (~np.isnan(mat)).sum(axis=1).astype(float)
    with warnings.catch_warnings():
        # 整行都缺失的基因，nanmean/nanvar 会喊 "Mean of empty slice"；
        # 那正是我们要的结果（测不了 → NaN），不必刷屏。
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(mat, axis=1)
        var = np.nanvar(mat, axis=1, ddof=1)  # 无偏方差；可用样本 < 2 时是 NaN
    return n, mean, var


def _welch(mat_a: np.ndarray, mat_b: np.ndarray) -> _Welch:
    """一次算完所有基因的 Welch 两样本统计量（NaN 跳过，不当 0）。

    mat_a、mat_b 是 (n_genes, n_samples) 矩阵，列里可以有 NaN。
    """
    n_a, mean_a, var_a = _group_stats(mat_a)
    n_b, mean_b, var_b = _group_stats(mat_b)

    with np.errstate(divide="ignore", invalid="ignore"):
        se2 = var_a / n_a + var_b / n_b          # 差值标准误的平方
        se = np.sqrt(se2)                        # 差值标准误（log2 轴上就是 log2FC 的 se）
        t_stat = (mean_b - mean_a) / se          # t 统计量（也是 z 统计量，两者只差参考分布）
        # Welch–Satterthwaite 自由度：两组方差不等时不能按 n_a+n_b-2 算
        df = se2 ** 2 / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))

    if _HAS_SCIPY:
        with np.errstate(invalid="ignore"):
            p = 2.0 * _stats.t.sf(np.abs(t_stat), df)
    else:
        # 正态近似 fallback：t 统计量不变，只是把参考分布从 t(df) 换成 N(0,1)。
        # 双尾 p = 2 * (1 - Φ(|z|))，Φ 用误差函数 erf 算。
        _erf = np.vectorize(math.erf)
        p = 2.0 * (1.0 - 0.5 * (1.0 + _erf(np.abs(t_stat) / math.sqrt(2.0))))

    # 两组方差都为 0（se==0）时 t 统计量无定义：均值相同 → 没有差异证据（p=1）；
    # 均值不同 → 两组是彻底分开的（p=0）。
    p = np.where(se2 == 0, np.where(mean_b == mean_a, 1.0, 0.0), p)
    # 某组可用样本 < 2 → 方差估不出来，这个基因在统计上"测不了"，p 记 NaN。
    # NaN 不会进 BH 的 m，也不会被悄悄当成"不显著"混进结果里。
    p = np.where((n_a < 2) | (n_b < 2), np.nan, p)
    return _Welch(n_a, n_b, mean_a, mean_b, var_a, var_b, se, df, p)


def _critical_value(df: np.ndarray, ci_level: float) -> np.ndarray:
    """置信区间用的双侧分位点：有 scipy 用 t(df)（小样本才准），没有就用正态近似。"""
    alpha = 1.0 - ci_level
    if _HAS_SCIPY:
        with np.errstate(invalid="ignore"):
            return _stats.t.ppf(1.0 - alpha / 2.0, df)
    return np.full(df.shape, _NORMAL.inv_cdf(1.0 - alpha / 2.0))


def differential_expression(
    expr_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    fc_threshold: float = 1.0,
    fdr_threshold: float = 0.05,
    input_scale: str = "log2",
    filter_quantile: Optional[float] = 0.25,
    min_group_n: int = 2,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """对每个基因做两组差异表达分析。

    参数:
        expr_df:     表达矩阵，第一列是基因名，其余列是样本；缺失的格子是 NaN（留空）
        metadata_df: 样本信息表，含 sample_id 和分组列
        group_col:   分组列名（如 "condition"）
        group_a:     A 组名
        group_b:     B 组名
        fc_threshold:  log2FC 阈值，默认 |log2FC| > 1（即差 2 倍）
        fdr_threshold: FDR 阈值，默认 < 0.05
        input_scale:  表达矩阵的值空间："log2"（默认，如 log2(FPKM+1)）或 "linear"
        filter_quantile: 独立过滤的分位阈值，默认 0.25（换成 None 关闭过滤）。
                         门槛 = 全样本平均表达的第 filter_quantile 分位数；
                         两组均值都低于门槛的基因在检验前剔除，且不计入 BH 的 m
        min_group_n:  每组至少要有多少个可用样本才做检验，默认 2（算方差的下限）
        ci_level:     log2FC 置信区间的置信水平，默认 0.95

    返回:
        DataFrame，每行一个基因，含
        gene / mean_a / mean_b / log2fc / log2fc_ci_low / log2fc_ci_high / se /
        pvalue / fdr / significant。
        log2fc 的方向约定为 **B 相对 A**：正值 = 该基因在 group_b 中更高。
        表里只有"真正参与检验"的基因；被过滤掉多少读 res.attrs：
        n_genes_total / n_genes_tested / n_genes_filtered /
        n_genes_filtered_low_expression / n_genes_filtered_unusable /
        filter_threshold / bh_m（写在 .attrs 里是为了不破坏列契约——
        nodes.py 等调用方只按列名取值，不需要改）。
    """
    # 1. 从 metadata 取出两组各自的样本名
    meta = metadata_df.set_index("sample_id")
    samples_a = meta[meta[group_col] == group_a].index.tolist()
    samples_b = meta[meta[group_col] == group_b].index.tolist()
    if not samples_a or not samples_b:
        raise ValueError(
            f"分组为空：{group_a} 有 {len(samples_a)} 个样本、{group_b} 有 {len(samples_b)} 个；"
            f"检查 metadata 的 {group_col} 列里是否真的有这两个取值"
        )

    # 2. 表达矩阵：第一列是基因名，设成 index，剩下全是样本
    expr = expr_df.copy()
    expr = expr.set_index(expr.columns[0])

    # 3. 向量化：一次取出两组矩阵（逐基因 for + .loc 在真实数据上要跑两分多钟）。
    #    to_numpy 遇到空单元格（缺失）会得到 NaN，后面的统计全走 nan 语义。
    genes = list(expr.index)
    mat_a = expr[samples_a].to_numpy(dtype=float)  # (n_genes, n_a)
    mat_b = expr[samples_b].to_numpy(dtype=float)  # (n_genes, n_b)

    w = _welch(mat_a, mat_b)

    # log2 fold change，方向是 B 相对 A（正数 = 在 group_b 中更高）。
    # 值空间必须先约定：TCGA 数据是 log2(FPKM+1)，已经在对数轴上，相减就是 log2FC；
    # 只有线性量纲才用 log2(比值)。对 log2 数据再取 log2(比值) 是双重取对数，
    # 结果没有倍数语义，且依赖 log 的平移量——同一个基因换个伪计数就会变成另一个数。
    if input_scale == "linear":
        with np.errstate(divide="ignore", invalid="ignore"):
            log2fc = np.log2((w.mean_b + 1.0) / (w.mean_a + 1.0))
            # 线性量纲下 log2FC 是"比值的对数"，Welch 的 se（差值标准误）不能直接搬过来。
            # 用 delta 方法把 g(x)=log2(x+1) 的一阶近似方差落到两组均值上：
            #   Var(log2FC) ≈ [ Var_a/n_a/(mean_a+1)^2 + Var_b/n_b/(mean_b+1)^2 ] / (ln2)^2
            se_fc = np.sqrt(
                w.var_a / w.n_a / (w.mean_a + 1.0) ** 2
                + w.var_b / w.n_b / (w.mean_b + 1.0) ** 2
            ) / math.log(2.0)
    else:
        log2fc = w.mean_b - w.mean_a
        # log2 轴上"均值之差"就是 log2FC，Welch 的 se 直接就是它的标准误，
        # 不需要任何近似：CI = log2FC ± t(1-α/2, df) × se。
        se_fc = w.se

    # 4. 独立过滤（independent filtering）
    #    门槛取"全样本平均表达"分布的低分位；两组均值都低于门槛的基因，在两组里都基本没表达，
    #    测不出稳定差异，却照样占掉 BH 的一个 m 名额。
    #    判定只看表达量、不看 p 值——"独立"就是这个意思；若按 p 值挑基因，剩下的基因做 BH
    #    时 m 已经被人为缩小，FDR 会失真。
    with warnings.catch_warnings():
        # 整份数据都缺失时 nanmean/nanquantile 会喊 "All-NaN slice"，
        # 结果（nan 门槛 → 全部不可检验）本身就是我们要的答案。
        warnings.simplefilter("ignore", RuntimeWarning)
        overall_mean = np.nanmean(np.hstack([mat_a, mat_b]), axis=1)
        if filter_quantile is None:
            threshold = float("nan")
        else:
            threshold = float(np.nanquantile(overall_mean, filter_quantile))
    if filter_quantile is None:
        low_expr = np.zeros(len(genes), dtype=bool)
    else:
        low_expr = (w.mean_a < threshold) & (w.mean_b < threshold)

    # 不可检验的基因：某组可用样本不足，或两组都完全没有值。它们连"低表达"都谈不上，
    # 单独计数（优先归到这一类，避免和低表达重复计一个人头）。
    unusable = (w.n_a < min_group_n) | (w.n_b < min_group_n)
    keep = ~low_expr & ~unusable
    idx = np.flatnonzero(keep)

    n_total = len(genes)
    n_unusable = int(unusable.sum())
    n_low = int((low_expr & ~unusable).sum())
    n_tested = int(idx.size)

    genes_k = [genes[i] for i in idx]
    log2fc_k = log2fc[idx]
    se_k = se_fc[idx]

    # 5. log2FC 的置信区间：点估计 ± 分位点 × 标准误。
    #    se==0（两组都零方差）时分位点会是 NaN（自由度 0/0），直接给 [log2fc, log2fc]：
    #    那种基因在两组里是完全分开的两条水平线，区间确实退化成一个点。
    with np.errstate(invalid="ignore"):
        crit = _critical_value(w.df[idx], ci_level)
        half = np.where(se_k == 0, 0.0, crit * se_k)

    res = pd.DataFrame({
        "gene": genes_k,
        "mean_a": w.mean_a[idx],
        "mean_b": w.mean_b[idx],
        "log2fc": log2fc_k,
        "log2fc_ci_low": log2fc_k - half,
        "log2fc_ci_high": log2fc_k + half,
        "se": se_k,
        "pvalue": w.pvalue[idx],
    })

    # 6. BH 多重检验校正：按 p 值从小到大排序，逐步收紧阈值。
    #    m 只能是"真正参与了检验的基因数"：被独立过滤剔除的基因已经不在表里，
    #    p 值为 NaN 的（样本不足）也不计入。m 算错会让所有 FDR 系统性偏大或偏小。
    res = res.sort_values("pvalue", na_position="last").reset_index(drop=True)
    finite = res["pvalue"].notna().to_numpy()
    m = int(finite.sum())
    fdr = np.full(len(res), np.nan)
    if m:
        p = res.loc[finite, "pvalue"].to_numpy(dtype=float)
        adj = p * m / (np.arange(m) + 1.0)
        # BH 要求 FDR 单调不减，从后往前取最小值
        fdr[finite] = np.minimum.accumulate(adj[::-1])[::-1]
        fdr = np.clip(fdr, None, 1.0)
    res["fdr"] = fdr

    # 7. 显著：FDR 够小 且 变化倍数够大（NaN 比较结果为 False，自动不算显著）
    res["significant"] = (res["fdr"] < fdr_threshold) & (res["log2fc"].abs() > fc_threshold)

    res = res.sort_values("fdr", na_position="last").reset_index(drop=True)

    # 8. 过滤计数随表返回：调用方只按列取值，不会受影响；要写报告/画图的地方
    #    读 res.attrs 就能说清"到底剔了多少个基因、门槛是多少"。
    res.attrs.update({
        "n_genes_total": n_total,
        "n_genes_tested": n_tested,
        "n_genes_filtered": n_total - n_tested,
        "n_genes_filtered_low_expression": n_low,
        "n_genes_filtered_unusable": n_unusable,
        "filter_threshold": threshold,
        "filter_quantile": filter_quantile,
        "bh_m": m,
        "ci_level": ci_level,
    })
    return res
