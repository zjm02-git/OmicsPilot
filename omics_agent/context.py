"""上下文预算层：给送进 LLM 的输入设一个"硬上限"。

背景（实测）：interpret_node 把差异表 + 富集结果 + 基因功能全量拼进 prompt，
qa_node 把 5 张卡片各截 500 字拼进 prompt。样本一多、显著基因一多，
prompt 会悄悄顶到模型上下文上限：被截断的是结尾（恰好是最需要解读的证据），
而 token 账单照付。这里把"截断"从隐式变成显式、可度量。

做法：每个证据块带 priority，超预算时从**低优先级**开始整块丢弃，
丢掉的块在输出里留一行显式占位（让 LLM 知道自己没看全，不要在报告里假装有证据）。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# 中日韩文字：1 个字≈1 token；其余字符（英文/数字/符号）≈3.5 字/token。
# 这是估算，不是 tokenizer——目的是"预算可控"，误差 10% 量级可以接受。
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")


def _count_cjk(text: str) -> int:
    return len(_CJK_RE.findall(text))


def estimate_tokens(text: str) -> int:
    """粗略估算一段文本的 token 数（保守偏高，宁可多留余量）。"""
    if not text:
        return 0
    cjk = _count_cjk(text)
    other = len(text) - cjk
    return int(cjk + other / 3.5) + 1


def truncate_tokens(text: str, max_tokens: int, ellipsis: str = "…") -> str:
    """把单块文本截到 max_tokens 以内（按比例切，末尾补省略号）。"""
    if not text or max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    ratio = max_tokens / float(estimate_tokens(text))
    cut = max(1, int(len(text) * ratio * 0.95))
    out = text[:cut]
    # 估算有误差，循环收缩直到真的进预算（最多几轮，长度单调下降保证能退出）
    while out and estimate_tokens(out + ellipsis) > max_tokens and len(out) > 1:
        out = out[: max(1, int(len(out) * 0.9))]
    return out + ellipsis


def fit_blocks(blocks: Iterable[dict[str, Any]], max_tokens: int) -> tuple[dict[str, str], dict[str, Any]]:
    """按优先级把多个证据块塞进 token 预算。

    blocks: [{"name": "差异表", "text": "...", "priority": 3}, ...]，priority 越大越不能丢。
    返回 (name -> 文本, 统计信息)；被完全丢弃的块文本为空字符串。
    """
    items = [
        {"name": str(b.get("name") or f"block{i}"),
         "text": str(b.get("text") or ""),
         "priority": float(b.get("priority", 0))}
        for i, b in enumerate(blocks)
    ]
    budget = max(0, int(max_tokens))

    for item in items:
        item["tokens"] = estimate_tokens(item["text"])

    total = sum(i["tokens"] for i in items)
    dropped: list[str] = []
    truncated: list[str] = []

    # ① 整块丢弃：从低优先级开始，直到进预算
    if total > budget:
        for item in sorted(items, key=lambda x: (x["priority"], -items.index(x))):
            if total <= budget:
                break
            if item["tokens"] == 0:
                continue
            # 只剩最后一个块时别再丢：唯一的证据被整块丢掉，等于让模型"无证据作答"，
            # 交给下面第 ② 步按比例截断，至少保留主干。
            if sum(1 for i in items if i["tokens"] > 0) <= 1:
                break
            total -= item["tokens"]
            item["tokens"] = 0
            item["text"] = ""
            dropped.append(item["name"])

    # ② 仍超预算（比如只剩一个巨块）：按比例削它，直到进预算
    if total > budget:
        for item in items:
            if item["tokens"] == 0:
                continue
            share = max(0, int(budget * (item["tokens"] / max(1, total))))
            if share < item["tokens"]:
                item["text"] = truncate_tokens(item["text"], share)
                item["tokens"] = estimate_tokens(item["text"])
                truncated.append(item["name"])
        total = sum(i["tokens"] for i in items)

    texts = {i["name"]: i["text"] for i in items}
    info = {
        "budget_tokens": budget,
        "used_tokens": total,
        "dropped": dropped,
        "truncated": truncated,
    }
    return texts, info


def drop_note(info: dict[str, Any]) -> str:
    """被裁掉的部分要在 prompt 里明说，避免模型"假装看到了全部证据"。"""
    parts = []
    if info.get("dropped"):
        parts.append("、".join(info["dropped"]) + "（超出上下文预算，已省略）")
    if info.get("truncated"):
        parts.append("、".join(info["truncated"]) + "（已截断到预算内）")
    if not parts:
        return ""
    return "【注意】以下证据因上下文预算不足已做裁剪：" + "；".join(parts) + "。请在结论中说明证据可能不完整。\n\n"
