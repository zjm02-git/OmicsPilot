"""LLM 零样本重排：用 DeepSeek 给候选卡片重新打分，替代 cross-encoder。

为什么需要（生活比喻）：
  向量和 BM25 两路融合后，各自分数尺度不同，融合结果仍可能把「次相关」
  排在前面。让 LLM 亲眼看一遍「问题 + 每张候选卡片」，按相关性重新排一遍，
  相当于请个专家把候选名单再核一遍。

为什么不用 cross-encoder（rerank 模型的传统做法）：
  cross-encoder 要单独下载一个小模型跑本地；这里复用现有 DeepSeek，
  零新增依赖、零新模型，面试还更好讲（"用 LLM 做零样本重排"）。

降级约定：LLM 挂了或输出解析不了，就原样返回候选顺序——宁可不重排，不能更差。
"""

import json
import re

from langchain_core.messages import HumanMessage

from omics_agent.config import get_model
from omics_agent.resilience import llm_call

# 每张卡片喂给 LLM 时截断的长度：判断相关度只看前半段就够，省 token
_DOC_SNIPPET_CHARS = 400


def rerank(question: str, candidates, top_k: int = 5):
    """用 LLM 对候选卡片重排，返回前 top_k 个（失败则原序返回）。

    参数：
      question   用户问题
      candidates [(基因名, 原文, 分数), ...] 已按某种方式粗排的候选列表
      top_k      重排后返回几个

    返回：仍是 [(基因名, 原文, 分数), ...] 三元组，顺序是 LLM 判定的相关度。
    """
    if len(candidates) <= 1:
        return candidates[:top_k]

    # 给候选编上 1 起的号，拼进 prompt
    lines = [f"[{i}] {gene}: {doc[:_DOC_SNIPPET_CHARS]}" for i, (gene, doc, _) in enumerate(candidates, start=1)]
    docs_text = "\n".join(lines)

    prompt = (
        "你是一名相关性重排器。根据用户问题，判断下面候选基因卡片与问题的相关程度，"
        "按从高到低输出卡片编号的 JSON 数组（例如 [3, 1, 5, 2, 4]）。"
        "只输出 JSON 数组本身，不要任何解释或多余文字。\n\n"
        f"【用户问题】\n{question}\n\n【候选卡片】\n{docs_text}"
    )

    try:
        res = llm_call(get_model(), [HumanMessage(prompt)], timeout_s=60.0, attempts=2)
        order = _parse_order(res.content, len(candidates))
    except Exception:  # noqa: BLE001 - 重排只是锦上添花，失败退回原顺序
        order = list(range(len(candidates)))

    return [candidates[i] for i in order][:top_k]


def _parse_order(text: str, n: int) -> list[int]:
    """从 LLM 输出里抽出 JSON 数组并校验，缺失/越界的编号补到末尾。

    返回 0 起的下标列表；解析不出或全越界就抛异常，让调用方回退原序。
    """
    m = re.search(r"\[[\d,\s]+\]", text or "")
    if not m:
        raise ValueError("输出里没找到 JSON 数组")
    arr = json.loads(m.group(0))

    order: list[int] = []
    for x in arr:
        i = int(x) - 1            # 模型给的是 1 起编号，转 0 起下标
        if 0 <= i < n and i not in order:
            order.append(i)
    if not order:
        raise ValueError("数组为空或全部越界")

    # 模型若漏了某些编号，把它们补到末尾，保证候选不丢
    for i in range(n):
        if i not in order:
            order.append(i)
    return order
