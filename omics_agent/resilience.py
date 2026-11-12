"""韧性层：LLM 调用的超时、重试、退避、熔断，以及"结构化输出失败"的自纠错与规则兜底。

三个真实故障，这个文件逐一兜住（对应面试里"模型/工具调用失败怎么办"那一问）：
  1. 模型网关抖动（超时 / 502 / 限流）→ 指数退避 + 抖动重试；超时用线程池硬控，
     不依赖模型客户端是否支持 per-request timeout；
  2. 网关持续不可用 → 熔断器打开，快速失败，避免每个请求都等满超时、
     把工作线程拖死（这是"雪崩"最常见的起点）；
  3. 结构化输出（with_structured_output）失败 → 让模型看到"上次不合格"再试一次，
     仍失败则退回关键词规则：宁可路由保守一点，也不让整条流水线崩在第一个节点。
"""

from __future__ import annotations

import concurrent.futures
import random
import threading
import time
from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage

from omics_agent import metrics

# 模型调用是 IO 密集且可能很慢（解读节点实测 ~18s），用独立线程池做超时控制。
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm-io")

_CORRECTION_MESSAGE = HumanMessage(
    "上一次调用没有返回可解析的结构化结果。请严格按给定的 JSON Schema 只输出一个 JSON 对象，"
    "不要输出解释、不要输出 markdown 代码块。"
)


class CircuitOpenError(RuntimeError):
    """熔断打开：不发起真实调用，直接失败，让上层走兜底路径。"""


class CircuitBreaker:
    """极简三态熔断器：closed → open → half_open → closed。"""

    def __init__(self, name: str = "llm", failure_threshold: int = 5, reset_timeout: float = 30.0) -> None:
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.reset_timeout = float(reset_timeout)
        self._lock = threading.Lock()
        self._state = "closed"
        self._failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    def allow(self) -> bool:
        """是否放行这次调用；open 且冷却未到时返回 False。"""
        with self._lock:
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self.reset_timeout:
                    self._state = "half_open"  # 放一个探针出去试探
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "half_open" or self._failures >= self.failure_threshold:
                if self._state != "open":
                    metrics.incr("breaker_open_total")
                self._state = "open"
                self._opened_at = time.monotonic()


def invoke_with_timeout(model: Any, messages: Sequence[Any], timeout_s: float = 90.0) -> Any:
    """给模型调用套一个硬超时（默认 90s）。

    为什么不在客户端构造函数里设 timeout：不同调用点的合理上限不同
    （意图解析 30~45s、结果解读 120s），做在调用边界最省事，也不依赖具体集成实现。
    """
    timeout_s = float(timeout_s)
    future = _EXECUTOR.submit(model.invoke, messages)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        metrics.incr("llm_timeout_total")
        raise TimeoutError(f"LLM 调用超时（>{timeout_s:.0f}s）") from exc


def call_with_retry(
    fn: Callable[[], Any],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 6.0,
    jitter: float = 0.3,
    breaker: CircuitBreaker | None = None,
    sleep: Callable[[float], None] = time.sleep,
    label: str = "call",
) -> Any:
    """指数退避 + 随机抖动重试。只有最后一次仍失败才抛原始异常。"""
    attempts = max(1, int(attempts))
    last_exc: Exception | None = None

    for i in range(attempts):
        if breaker is not None and not breaker.allow():
            metrics.incr(f"{label}_short_circuit_total")
            raise CircuitOpenError(f"{label}: 熔断中，跳过本次调用")

        try:
            out = fn()
        except CircuitOpenError:
            raise
        except Exception as exc:  # noqa: BLE001 - 网关异常类型不可枚举，统一重试
            last_exc = exc
            if breaker is not None:
                breaker.record_failure()
            metrics.incr(f"{label}_fail_total")
            if i >= attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** i))
            delay *= 1.0 + random.uniform(-jitter, jitter)
            metrics.incr(f"{label}_retry_total")
            sleep(max(0.0, delay))
            continue
        else:
            if breaker is not None:
                breaker.record_success()
            return out

    metrics.incr(f"{label}_exhausted_total")
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label}: 未知失败")


def llm_call(
    model: Any,
    messages: Sequence[Any],
    *,
    timeout_s: float = 90.0,
    attempts: int = 3,
    breaker: CircuitBreaker | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """带超时 + 重试的普通模型调用（interpret / qa 节点用）。"""
    return call_with_retry(
        lambda: invoke_with_timeout(model, messages, timeout_s),
        attempts=attempts,
        breaker=breaker,
        sleep=sleep,
        label="llm",
    )


def structured_output_call(
    model: Any,
    schema: Any,
    messages: Sequence[Any],
    *,
    timeout_s: float = 45.0,
    attempts: int = 2,
    breaker: CircuitBreaker | None = None,
    fallback: Callable[[Exception], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """结构化输出：失败先"自纠错"一次，仍失败则走规则兜底（fallback）。"""
    msgs: list[Any] = list(messages)
    last_exc: Exception | None = None

    for attempt in range(max(1, int(attempts))):
        if breaker is not None and not breaker.allow():
            last_exc = CircuitOpenError("llm_structured: 熔断中，跳过调用")
            metrics.incr("llm_structured_short_circuit_total")
            break
        try:
            structured = model.with_structured_output(schema)
            out = invoke_with_timeout(structured, msgs, timeout_s)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if breaker is not None:
                breaker.record_failure()
            metrics.incr("llm_structured_fail_total")
            if attempt >= attempts - 1:
                break
            metrics.incr("llm_structured_retry_total")
            msgs = list(messages) + [_CORRECTION_MESSAGE]
            sleep(min(4.0, 0.5 * (2 ** attempt)))
            continue
        else:
            if breaker is not None:
                breaker.record_success()
            return out

    if fallback is None:
        raise last_exc if last_exc is not None else RuntimeError("llm_structured: 未知失败")
    metrics.incr("llm_structured_fallback_total")
    return fallback(last_exc if last_exc is not None else RuntimeError("unknown"))


# ── 规则兜底：模型不可用时，用关键词把需求路由到最可能的分析类型 ──
# 顺序即优先级（同分取靠前者）：差异分析是主路径，其它是支路。
_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("differential", ("差异", "上调", "下调", "log2fc", "log2fold", "fold change", "火山图",
                      "differential", "deg", "显著基因")),
    ("pca", ("pca", "主成分", "降维", "principal component", "variance")),
    ("clustering", ("聚类", "分群", "cluster", "kmeans", "k-means", "分组模式")),
    ("qa", ("哪些基因", "什么功能", "功能是", "参与", "通路", "注释", "介绍一下",
            "知识库", "相关基因", "什么是")),
)


def keyword_intent(text: str) -> str:
    """关键词路由：只用显式词表打分，不猜、不联网。命中为空时返回 "other"。"""
    lowered = (text or "").lower()
    best_type, best_score = "other", 0
    for intent, keywords in _INTENT_RULES:
        score = sum(1 for kw in keywords if kw in lowered)
        if score > best_score:
            best_type, best_score = intent, score
    return best_type
