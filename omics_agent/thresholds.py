"""显著差异基因的阈值约束：**唯一的校验真源**。

为什么单独抽一个模块（而不是散在 nodes.py / web.py / 前端各写一遍）：
  阈值同时被三处用到——
    ① HITL 节点把「允许范围」抛给前端渲染输入框；
    ② /confirm 收到用户填的值必须校验（前端校验能被绕过，后端才是真防线）；
    ③ 差异分析真正用来判定显著性的就是这两个数。
  三处各写一遍常量，迟早会漂移。漂移的表现是最坏的那种：**静默**——
  「前端允许填 3.0，后端默默按 1.0 跑」，用户以为换了个标准，其实没换。
  所以范围只在这里定义一次，别处一律引用。

阈值在真实研究里怎么定的（这段是给读代码的人看的口径，不是可选项）：
  · FDR（BH 校正后的 p 值）：**必须 ≤ 0.05**，这是差异表达分析的事实标准；
    0.01 用于严格筛选（如找标志基因），0.1 只用于探索性/样本量很小的场景。
  · |log2FC|：**≥ 1（2 倍）** 是常规门槛；0.585（1.5 倍）用于宽松筛选，
    2.0（4 倍）用于严格筛选。取 0 等于关掉倍数过滤，只按 FDR 判 —— 合法但少见。
  · 样本量少（每组 < 5）时统计功效低，通常放宽 FC、保留 FDR 不放宽，
    而不是反过来 —— 放宽 FDR 会直接放大假阳性。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ── |log2FC| 下限（fold change，取对数后）─────────────────────────────────
# 0.0 = 只按 FDR 判、不按倍数过滤；5.0 = 32 倍，再往上几乎筛不出基因。
FC_MIN = 0.0
FC_MAX = 5.0
FC_DEFAULT = 1.0

# ── FDR 上限（Benjamini-Hochberg 校正后的 p 值）───────────────────────────
# 0.001 是文献里能见到的最严格取值；0.5 以上已无统计意义（相当于扔硬币）。
FDR_MIN = 0.001
FDR_MAX = 0.5
FDR_DEFAULT = 0.05

# ── 推荐区间：绝大多数差异表达研究实际采用的取值 ─────────────────────────
# 前端把它显示成"建议值"提示，不强制；用户填区间内的其它值一律合法。
FC_RECOMMENDED = (0.585, 2.0)     # 1.5 倍 ~ 4 倍
FDR_RECOMMENDED = (0.01, 0.1)

# 前端输入框的步长：0.05 对应 1.07 倍，0.005 对应 FDR 的常用刻度
FC_STEP = 0.05
FDR_STEP = 0.005


@dataclass(frozen=True)
class ThresholdCheck:
    """一次校验的结果。ok=False 时 message 是直接给用户看的中文原因。"""

    ok: bool
    message: str = ""


def _number_error(label: str, value: Any, low: float, high: float) -> str:
    """把一个值按 [low, high] 校验；返回空串表示通过。"""
    # bool 是 int 的子类，True/False 会被当成 1/0 —— 必须显式挡掉，
    # 否则前端传个复选框状态进来就静默变成 fc=1.0。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{label} 必须是数字（收到 {value!r}）"
    number = float(value)
    if not math.isfinite(number):
        return f"{label} 必须是有限数字（收到 {value!r}）"
    if number < low or number > high:
        return f"{label} 需在 {low:g}–{high:g} 之间（收到 {number:g}）"
    return ""


def validate(fc_threshold: Any = None, fdr_threshold: Any = None) -> ThresholdCheck:
    """校验用户填的阈值。None 表示"这一项沿用当前值"，不参与校验。

    两个都填错时只报第一个，够用且比堆一串错误好读。
    """
    if fc_threshold is not None:
        error = _number_error("|log2FC| 下限", fc_threshold, FC_MIN, FC_MAX)
        if error:
            return ThresholdCheck(False, error)
    if fdr_threshold is not None:
        error = _number_error("FDR 上限", fdr_threshold, FDR_MIN, FDR_MAX)
        if error:
            return ThresholdCheck(False, error)
    return ThresholdCheck(True)


def bounds() -> dict[str, dict[str, Any]]:
    """给前端的输入框边界（min/max/step/默认值/建议区间）。

    一份数据同时喂给前端渲染和后端报错文案，保证"页面允许填的"与
    "后端接受的"永远是同一个范围。
    """
    return {
        "fc_threshold": {
            "label": "|log2FC| 下限",
            "min": FC_MIN,
            "max": FC_MAX,
            "step": FC_STEP,
            "default": FC_DEFAULT,
            "recommended": [FC_RECOMMENDED[0], FC_RECOMMENDED[1]],
            "help": "表达量差多少倍才算差异：1.0 = 2 倍，0.585 = 1.5 倍，2.0 = 4 倍；0 = 不按倍数过滤",
        },
        "fdr_threshold": {
            "label": "FDR 上限",
            "min": FDR_MIN,
            "max": FDR_MAX,
            "step": FDR_STEP,
            "default": FDR_DEFAULT,
            "recommended": [FDR_RECOMMENDED[0], FDR_RECOMMENDED[1]],
            "help": "BH 多重检验校正后的 p 值上限：0.05 是事实标准，0.01 更严格，0.1 仅用于探索",
        },
    }


def describe(thresholds: dict[str, Any] | None) -> str:
    """把一组阈值渲染成一句人话（写进报告，让读者知道按什么标准筛的）。"""
    data = thresholds or {}
    fc = data.get("fc_threshold", FC_DEFAULT)
    fdr = data.get("fdr_threshold", FDR_DEFAULT)
    try:
        fc_text = f"|log2FC| > {float(fc):g}"
        fdr_text = f"FDR < {float(fdr):g}"
    except (TypeError, ValueError):
        return ""
    return f"{fc_text} 且 {fdr_text}"
