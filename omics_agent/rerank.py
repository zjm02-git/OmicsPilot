"""LLM 零样本重排：用 DeepSeek 给候选卡片重新打分，替代 cross-encoder。

为什么需要（生活比喻）：
  向量和 BM25 两路融合后，各自分数尺度不同，融合结果仍可能把「次相关」
  排在前面。让 LLM 亲眼看一遍「问题 + 每张候选卡片」，按相关性重新排一遍，
  相当于请个专家把候选名单再核一遍。

为什么不用 cross-encoder（rerank 模型的传统做法）：
  cross-encoder 要单独下载一个小模型跑本地；这里复用现有 DeepSeek，
  零新增依赖、零新模型，面试还更好讲（"用 LLM 做零样本重排"）。

降级约定：LLM 挂了或输出解析不了，就原样返回候选顺序——宁可不重排，不能更差。

本次补的三件产品化的事（原来只有 rerank() 一个裸函数，且只有评测脚本在用）：
  1) 开关：RERANK_ENABLED 默认 0（关）。重排要多一次 LLM 往返（约 1~3s），
     本质是「拿延迟换精度」，应该由部署方按场景打开，而不是默默给每个
     用户都加上——这也是它原来没接进产品链路的原因。
  2) 缓存：同一句提问在同一进程里只重排一次（LRU + TTL）。重排是纯函数
     （问题 + 候选文本 -> 顺序），缓存天然安全；产品里同一句追问很常见。
  3) 降级：开关关 / 候选太少 / LLM 报错 / 输出解析失败，四种情况都原样
     返回粗排结果，并把原因写进 info——调用方能区分「没开重排」和
     「开了但退化了」，评测里也就不会把降级当成一次正常重排。
"""

import hashlib
import json
import os
import re
import time
from collections import OrderedDict

from langchain_core.messages import HumanMessage

from omics_agent.config import get_model
from omics_agent.resilience import llm_call

# 每张卡片喂给 LLM 时截断的长度：判断相关度只看前半段就够，省 token
_DOC_SNIPPET_CHARS = 400

# ── 本次重排的四种结局（写进 info["status"]，评测/日志里可统计）──
STATUS_OK = "ok"                    # 真的重排了
STATUS_CACHE_HIT = "cache_hit"      # 命中缓存，等价于重排结果
STATUS_DISABLED = "disabled"        # 开关没开
STATUS_FEW = "few_candidates"       # 候选 <= 1，排不排都一样
STATUS_LLM_ERROR = "llm_error"      # 调用失败/超时/熔断
STATUS_PARSE_ERROR = "parse_error"  # 输出解析不出合法顺序

# ── 缓存（进程内 LRU + TTL）──
# 为什么不用磁盘缓存：重排结果依赖模型版本，跨进程复用容易「拿旧模型的答案」；
# 单进程 LRU 就能吃掉产品里的重复提问，且进程重启即失效，语义干净。
_CACHE = OrderedDict()   # key -> (写入时间戳, 顺序下标列表)
_CACHE_MAX = 256         # 最多缓 256 条提问
_CACHE_TTL_S = 1800.0    # 30 分钟过期：模型可能被换掉，旧序不过期就会一直生效


def rerank_enabled(enabled=None) -> bool:
    """开关判定：显式参数 > 环境变量 RERANK_ENABLED > 默认关。

    为什么每次现读环境变量而不是 import 时定值：
      部署想开就开（改 .env 重启即可），评测也能在同一进程里做 A/B
      （临时 os.environ["RERANK_ENABLED"]="1"），不必重新 import 模块。
    """
    if enabled is not None:
        return bool(enabled)
    return os.getenv("RERANK_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")


def _build_prompt(question: str, candidates) -> str:
    """拼 prompt：候选卡片带上 1 起的编号，让模型只需要回答编号顺序。"""
    lines = [
        f"[{i}] {gene}: {str(doc)[:_DOC_SNIPPET_CHARS]}"
        for i, (gene, doc, _) in enumerate(candidates, start=1)
    ]
    docs_text = "\n".join(lines)
    return (
        "你是一名相关性重排器。根据用户问题，判断下面候选基因卡片与问题的相关程度，"
        "按从高到低输出卡片编号的 JSON 数组（例如 [3, 1, 5, 2, 4]）。"
        "只输出 JSON 数组本身，不要任何解释或多余文字。\n\n"
        f"【用户问题】\n{question}\n\n【候选卡片】\n{docs_text}"
    )


def _llm_order(question: str, candidates) -> list:
    """问一次 LLM，返回 0 起的顺序下标。**会抛异常**——由调用方决定怎么降级。

    为什么把「调用」和「降级」拆开：只有让异常穿出来，上层才能区分
    「LLM 挂了」和「输出解析不了」这两种降级原因，降级才叫「明确」。
    """
    res = llm_call(get_model(), [HumanMessage(_build_prompt(question, candidates))],
                   timeout_s=60.0, attempts=2)
    return _parse_order(getattr(res, "content", ""), len(candidates))


def rerank(question: str, candidates, top_k: int = 5):
    """用 LLM 对候选卡片重排，返回前 top_k 个（失败则原序返回）。

    参数：
      question   用户问题
      candidates [(基因名, 原文, 分数), ...] 已按某种方式粗排的候选列表
      top_k      重排后返回几个

    返回：仍是 [(基因名, 原文, 分数), ...] 三元组，顺序是 LLM 判定的相关度。

    这是最早那个入口，保留它是因为评测脚本/脚本化调试需要一个「必定兜底、
    不会抛异常」的重排函数；产品链路请用下面的 maybe_rerank（有开关和缓存）。
    """
    if len(candidates) <= 1:
        return candidates[:top_k]

    try:
        order = _llm_order(question, candidates)
    except Exception:  # noqa: BLE001 - 重排只是锦上添花，失败退回原顺序
        order = list(range(len(candidates)))

    return [candidates[i] for i in order][:top_k]


def _cache_key(question: str, candidates, top_k: int) -> str:
    """缓存键：只哈希「LLM 真正看到的东西」（问题 + 卡片前 400 字及顺序）+ top_k。

    为什么不把粗排分数放进键：模型只看文本，不看分数；同一批候选、同样的
    文本顺序，重排结果必然相同，没必要因为浮点分数抖动就重新花钱问一次。
    """
    payload = [
        question,
        top_k,
        [[str(g), str(d)[:_DOC_SNIPPET_CHARS]] for g, d, _ in candidates],
    ]
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def _cache_get(key: str):
    """取缓存；过期就顺手删掉。命中时把该条挪到队尾（LRU 的「刚用过」）。"""
    hit = _CACHE.get(key)
    if hit is None:
        return None
    ts, order = hit
    if _CACHE_TTL_S > 0 and time.time() - ts > _CACHE_TTL_S:
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return list(order)


def _cache_put(key: str, order) -> None:
    """写缓存，超容量时从队头（最久没用过）开始淘汰。"""
    if _CACHE_MAX <= 0:
        return
    _CACHE[key] = (time.time(), list(order))
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def clear_cache() -> int:
    """清空缓存，返回清掉的条数。评测要跑 A/B 时先调一次，避免前一列污染后一列。"""
    n = len(_CACHE)
    _CACHE.clear()
    return n


def cache_size() -> int:
    """当前缓存条数（调试/测试用）。"""
    return len(_CACHE)


def _pack(results, info: dict, t0: float, return_info: bool):
    """统一出口：补上耗时，再按调用方要求决定要不要带上 info。"""
    info["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return (results, info) if return_info else results


def maybe_rerank(question: str, candidates, top_k: int = 5,
                 *, enabled=None, use_cache: bool = True, return_info: bool = False):
    """**上层一行调用**的产品入口：把「开关 + 缓存 + 降级」都收在这里。

    用法（接线是调度方的事，本模块不碰 nodes.py）：
        from omics_agent.rerank import maybe_rerank
        pool = rag.retrieve_hybrid(question, top_k=20)      # 先粗召回
        hits = maybe_rerank(question, pool, top_k=5)        # 再重排，形状完全一样

    参数：
      question   用户问题
      candidates [(基因名, 原文, 分数), ...]，形状与返回值一致，可直接替换
      top_k      返回几个
      enabled    显式开关；None = 读环境变量 RERANK_ENABLED（默认关）
      use_cache  是否用进程内缓存
      return_info True 时返回 (结果, info)，info 里写明这次走了哪条路

    返回：[(基因名, 原文, 分数), ...]，**永远不会因重排失败而变差**——
          任何一种降级都原样返回粗排前 top_k，并在 info 里标注原因。
    """
    info = {"status": STATUS_OK, "cache": "off", "n_candidates": len(candidates)}
    t0 = time.perf_counter()

    # ① 开关关着：直接返回粗排前 top_k（产品默认行为，不额外花一次 LLM）
    if not rerank_enabled(enabled):
        info["status"] = STATUS_DISABLED
        return _pack(candidates[:top_k], info, t0, return_info)

    # ② 候选 <= 1：重排不可能改变结果，省一次调用
    if len(candidates) <= 1:
        info["status"] = STATUS_FEW
        return _pack(candidates[:top_k], info, t0, return_info)

    # ③ 缓存
    key = _cache_key(question, candidates, top_k)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            info["status"] = STATUS_CACHE_HIT
            info["cache"] = "hit"
            return _pack([candidates[i] for i in cached][:top_k], info, t0, return_info)
        info["cache"] = "miss"

    # ④ 真的问 LLM；两类失败分开记，降级路径才「明确」
    try:
        order = _llm_order(question, candidates)
    except ValueError:
        # _parse_order 抛的是 ValueError：模型答了，但不是合法编号数组
        info["status"] = STATUS_PARSE_ERROR
        return _pack(candidates[:top_k], info, t0, return_info)
    except Exception:  # noqa: BLE001 - 超时/限流/熔断/网络，一律退回粗排
        info["status"] = STATUS_LLM_ERROR
        return _pack(candidates[:top_k], info, t0, return_info)

    if use_cache:
        _cache_put(key, order)
    return _pack([candidates[i] for i in order][:top_k], info, t0, return_info)


def _parse_order(text: str, n: int) -> list:
    """从 LLM 输出里抽出 JSON 数组并校验，缺失/越界的编号补到末尾。

    返回 0 起的下标列表；解析不出或全越界就抛 ValueError（不是别的异常），
    这样上层能把它和「网络失败」区分开。
    """
    m = re.search(r"\[[\d,\s]+\]", text or "")
    if not m:
        raise ValueError("输出里没找到 JSON 数组")
    arr = json.loads(m.group(0))

    order = []
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
