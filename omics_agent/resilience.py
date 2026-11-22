"""韧性层：LLM 调用的超时、重试、退避、熔断，以及"结构化输出失败"的自纠错与规则兜底。

三个真实故障，这个文件逐一兜住（对应面试里"模型/工具调用失败怎么办"那一问）：
  1. 模型网关抖动（超时 / 502 / 限流）→ 指数退避 + 抖动重试；
  2. 网关持续不可用 → 熔断器打开，快速失败，避免每个请求都等满超时把线程拖死；
  3. 结构化输出（with_structured_output）失败 → 让模型看到"上次不合格"再试一次，
     仍失败则退回关键词规则：宁可路由保守一点，也不让整条流水线崩在第一个节点。

本轮修掉了两个"看起来有、其实没有"的韧性（都是原实现里的真实缺陷，注释留证）：

  P1 线程池"硬超时"并不硬（原 resilience.py:91-97）
     原实现是 future = _EXECUTOR.submit(...); future.result(timeout); future.cancel()。
     两个事实让它失效：
     · future.cancel() 只对"还排在队列里、没被线程取走"的任务有效；对 RUNNING 的调用
       返回 False 且什么都不做 —— 模型 SDK 卡在一条长连接上时，那个线程永远不归还；
     · ThreadPoolExecutor 的队列是无界的：8 个线程被占满后，后来的 submit() 会静静排队，
       每个调用都在自己的 future.result(timeout) 上等满 30~120s 才失败，
       调用方的线程、连接、内存全部被拖住 —— 这正是要防的雪崩。
     现在（见 bind_request_deadline / invoke_with_timeout）：
     · 首选把 deadline 下沉到真正能中断 IO 的 HTTP 客户端（per-request timeout +
       max_retries=0），调用本身可被打断，连接被关掉、线程立即归还；
     · 线程池降级为最后兜底，并用 BoundedSemaphore 把在途调用数硬性限住：
       拿不到名额立刻抛 PoolSaturatedError（毫秒级），而不是无界排队等满一个超时；
     · 名额在线程真正结束（future 完成回调）时才归还 —— 所以"线程被占死"会体现在
       名额枯竭上，后面的调用快速失败，而不是继续堆；
     · 兜底路径真的命中时，把 cancel() 没生效这件事记成指标 llm_executor_stuck_total，
       让"线程被占死"在监控里可见，而不是悄悄发生。

  P2 熔断器只在进程内（原 nodes.py:57 的模块级单例）
     3 个 uvicorn worker 就是 3 份独立计数，各开各的闸，等于把阈值放大了 3 倍。
     现在（见 CircuitBreaker）：失败计数可放 Redis（INCR + EXPIRE），得到跨进程语义；
     Redis 不可用时自动退回进程内计数并计指标，可用性不受影响。
     接线方式（在 web.py 里一行，本文件不 import web，避免循环依赖）：
         from omics_agent import resilience
         resilience.configure_breaker_backend(CACHE.redis)   # 或 set_breaker_redis(...)
     nodes.py 不需要改：CircuitBreaker 在每次操作时读模块级后端，而不是构造时固定，
     所以 import 期就建好的 LLM_BREAKER 也能吃到 web.py 后配的 Redis。

"现在的超时到底终止了什么"（本轮的核心结论，也是面试时该讲清的一句话）：
     · 只有客户端超时（HTTP 层 read/write/connect 超时）真正"打断"了调用：TCP 层掐断、
       SDK 抛异常、线程归还；
     · future.result(timeout) 终止的只是"等待"，被叫方还在跑；
     · future.cancel() 对 RUNNING 调用无效（返回 False），这就是旧实现的全部问题；
     · 兜底路径的"名额打满"终止的是"新的调用"（快速失败），而不是那个已经卡住的调用。
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import os
import random
import threading
import time
from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage

from omics_agent import metrics

LOGGER = logging.getLogger(__name__)


def _env_num(name: str, default: float, *, as_int: bool = False) -> Any:
    """读一个数字型环境变量：写错了就用默认值，绝不因为一个手抖的 env 让服务起不来。"""
    raw = os.getenv(name)
    try:
        value = float(raw) if raw not in (None, "") else float(default)
    except (TypeError, ValueError):
        LOGGER.warning("环境变量 %s=%r 不是数字，回退默认值 %s", name, raw, default)
        value = float(default)
    return int(value) if as_int else value


# ── 线程池兜底通道：线程数是"同时能被卡住的数量"，名额数是"最多允许多少个调用在途" ──
# 两者分开的道理：名额 = 线程 + 有限排队，队列有界才能让"排不上"在毫秒级失败。
_POOL_WORKERS = max(1, _env_num("OMICS_LLM_IO_WORKERS", 8, as_int=True))
_QUEUE_SLOTS = max(0, _env_num("OMICS_LLM_QUEUE_SLOTS", 4, as_int=True))
_MAX_INFLIGHT = _POOL_WORKERS + _QUEUE_SLOTS
# 拿名额的宽限：默认 0 = 纯快速失败（真·不排队），需要容忍瞬时抖动时调大。
_ADMIT_GRACE_S = max(0.0, float(_env_num("OMICS_LLM_ADMIT_GRACE_S", 0.0)))

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_POOL_WORKERS, thread_name_prefix="llm-io"
)
_ADMISSION = threading.BoundedSemaphore(_MAX_INFLIGHT)

_STATS_LOCK = threading.Lock()
_INFLIGHT = 0          # 已拿到名额、还没结束的调用数（含正卡在 SDK 里的）
_PEAK_QUEUE_DEPTH = 0  # 排队深度（在途 - 线程数）的历史峰值，给健康检查/压测看

_CORRECTION_MESSAGE = HumanMessage(
    "上一次调用没有返回可解析的结构化结果。请严格按给定的 JSON Schema 只输出一个 JSON 对象，"
    "不要输出解释、不要输出 markdown 代码块。"
)


class CircuitOpenError(RuntimeError):
    """熔断打开：不发起真实调用，直接失败，让上层走兜底路径。"""


class PoolSaturatedError(RuntimeError):
    """本地容量饱和：线程池名额已满，直接快速失败（区别于"网关故障"）。

    为什么单独一个类型：本地拥塞和网关故障的处置完全不同。拥塞要退避重试（等在途调用
    还名额），不该记熔断失败 —— 否则本地一拥塞就把熔断闸门自己拉下来，陷入自我实现的雪崩。
    """


# ── 可观测：线程池/名额/超时的状态 ────────────────────────────
def pool_stats() -> dict[str, Any]:
    """当前兜底通道的状态快照（健康检查、压测、面试演示都用这个）。"""
    with _STATS_LOCK:
        inflight = _INFLIGHT
        peak = _PEAK_QUEUE_DEPTH
    return {
        "pool_workers": _POOL_WORKERS,
        "max_inflight": _MAX_INFLIGHT,
        "inflight": inflight,
        "queue_depth": max(0, inflight - _POOL_WORKERS),
        "peak_queue_depth": peak,
        "admits_grace_s": _ADMIT_GRACE_S,
    }


def _admit() -> None:
    """拿一个在途名额；拿不到就快速失败（这是 P1 的第二半：无界排队 → 有界拒绝）。"""
    global _INFLIGHT, _PEAK_QUEUE_DEPTH
    if _ADMIT_GRACE_S > 0:
        ok = _ADMISSION.acquire(timeout=_ADMIT_GRACE_S)
    else:
        ok = _ADMISSION.acquire(blocking=False)
    if not ok:
        metrics.incr("llm_pool_saturated_total")
        with _STATS_LOCK:
            inflight, peak = _INFLIGHT, _PEAK_QUEUE_DEPTH
        raise PoolSaturatedError(
            f"LLM 调用并发已满（在途 {inflight}/{_MAX_INFLIGHT}，峰值排队 {peak}），"
            "已快速失败而不是排队等满超时"
        )
    with _STATS_LOCK:
        _INFLIGHT += 1
        depth = max(0, _INFLIGHT - _POOL_WORKERS)
        if depth > _PEAK_QUEUE_DEPTH:
            _PEAK_QUEUE_DEPTH = depth
    metrics.incr("llm_pool_admitted_total")


def _release() -> None:
    global _INFLIGHT
    with _STATS_LOCK:
        if _INFLIGHT > 0:
            _INFLIGHT -= 1
    try:
        _ADMISSION.release()
    except ValueError:  # 释放多了只会是逻辑 bug，别把它变成调用方的异常
        LOGGER.debug("BoundedSemaphore 释放次数超过获取次数", exc_info=True)


# ── 客户端超时：把 deadline 放到真正能中断 IO 的一层 ────────────
_TIMEOUT_EXC_NAMES = frozenset(
    {
        "TimeoutError",
        "TimeoutException",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "APITimeoutError",
    }
)


def _is_timeout_exc(exc: BaseException) -> bool:
    """不 import openai/httpx 也认得出"客户端超时"：按类名在 MRO 里找。

    这样 resilience 不绑定具体集成和版本（openai、httpx、anthropic 的类名各不相同），
    集成换了也不用改这里的判断。
    """
    return any(klass.__name__ in _TIMEOUT_EXC_NAMES for klass in type(exc).__mro__)


def _httpx_timeout(timeout_s: float) -> Any:
    """per-request 的超时对象：connect/pool 卡死比 read 卡死少见，给一点余量即可。"""
    try:
        import httpx  # 由 langchain-deepseek → openai 间接安装，这里延迟引入

        return httpx.Timeout(
            timeout_s, connect=min(timeout_s, 10.0), pool=min(timeout_s, 10.0)
        )
    except Exception:  # noqa: BLE001 - 没有 httpx 就退化成一个裸数字，openai SDK 也认
        return float(timeout_s)


@dataclasses.dataclass
class DeadlineBinding:
    """一次调用的"超时绑定"结果。

    mode == "client"  ：模型被换成了带 per-request 超时的副本，调用真的可中断；
    mode == "executor"：集成没有可直接改的 HTTP 客户端（比如测试替身、其他厂商 SDK），
                        只能靠线程池兜底 —— 这时会记 llm_client_timeout_unavailable_total，
                        线上看到它就说明"这一路超时并不硬"，值得补 timeout 参数。
    """

    model: Any
    mode: str
    close: Callable[[], None]


def _noop() -> None:
    return None


def _close_quietly(bounded_root: Any, original_root: Any) -> None:
    """关掉我们自己造的客户端；如果 SDK 复用了原客户端，就绝对不能关（会砸到别人）。"""
    if bounded_root is None or getattr(bounded_root, "_client", None) is getattr(
        original_root, "_client", None
    ):
        return
    try:
        bounded_root.close()
    except Exception:  # noqa: BLE001 - 清理失败不影响本次调用的结论
        LOGGER.debug("关闭带超时的 HTTP 客户端失败", exc_info=True)


def bind_request_deadline(model: Any, timeout_s: float) -> DeadlineBinding:
    """把 timeout_s 绑到模型的 HTTP 客户端上，返回可中断的模型副本。

    为什么不用 model.model_copy(update={"timeout": timeout_s}) 这种事后改字段的写法：
    langchain 的模型在 validate_environment 阶段就把客户端造好了（而且每次 get_model()
    都是新实例），事后改字段不会重建客户端，看着设了超时其实没生效 —— 静默失效比
    不支持更坏。所以这里改的是客户端本身：OpenAI 兼容 SDK 的 with_options(timeout=...)
    是 per-request 生效的，而且顺手把 SDK 内部重试关掉（max_retries=0），
    否则 2 次内部重试会把 90s 硬超时悄悄放大成 ~270s。
    """
    timeout_s = max(0.001, float(timeout_s))
    root = getattr(model, "root_client", None)
    if root is None or not hasattr(root, "with_options"):
        metrics.incr("llm_client_timeout_unavailable_total")
        return DeadlineBinding(model, "executor", _noop)

    # 快路径：客户端本来就比要求的更紧、且内部重试已关 → 直接用原对象，省一次握手。
    # 这条路径也是"将来某天在 get_model() 里写了 timeout=" 时的自动优待。
    current = getattr(root, "timeout", None)
    if (
        isinstance(current, (int, float))
        and 0 < float(current) <= timeout_s
        and getattr(root, "max_retries", None) == 0
    ):
        metrics.incr("llm_client_timeout_reused_total")
        return DeadlineBinding(model, "client", _noop)

    try:
        bounded_root = root.with_options(timeout=_httpx_timeout(timeout_s), max_retries=0)
    except Exception:  # noqa: BLE001 - SDK 版本不支持/参数不认，退回线程池兜底
        metrics.incr("llm_client_timeout_failed_total")
        LOGGER.debug("with_options 绑不上 per-request 超时，退回线程池兜底", exc_info=True)
        return DeadlineBinding(model, "executor", _noop)

    bounded_model = _model_with_client(model, bounded_root)
    if bounded_model is None:
        _close_quietly(bounded_root, root)
        metrics.incr("llm_client_timeout_failed_total")
        return DeadlineBinding(model, "executor", _noop)

    metrics.incr("llm_client_timeout_applied_total")
    return DeadlineBinding(bounded_model, "client", lambda: _close_quietly(bounded_root, root))


def _model_with_client(model: Any, root_client: Any) -> Any | None:
    """造一个"同款模型、但挂着带超时客户端"的副本；做不到就返回 None（别硬来）。"""
    update: dict[str, Any] = {"root_client": root_client}
    completions = getattr(getattr(root_client, "chat", None), "completions", None)
    if completions is not None:
        update["client"] = completions
    try:
        if hasattr(model, "model_copy"):  # pydantic v2：langchain 的模型都是
            return model.model_copy(update=update)
        clone = model.__class__.__new__(model.__class__)  # pragma: no cover - 非 pydantic 分支
        clone.__dict__.update(getattr(model, "__dict__", {}))
        for key, value in update.items():
            setattr(clone, key, value)
        return clone
    except Exception:  # noqa: BLE001
        return None


def _invoke_in_pool(
    invoke: Callable[[], Any],
    timeout_s: float,
    *,
    on_done: Callable[[], None] | None = None,
) -> Any:
    """线程池兜底：名额 → 提交 → 限时等待。

    名额在线程真正结束时才归还（future 完成回调），而不是"等到超时就还" ——
    否则一个卡死的线程会让名额被反复借给新调用，队列又变回无界，P1 原地复活。
    """
    released = False

    def _cleanup(_future: Any = None) -> None:
        nonlocal released
        if not released:
            released = True
            _release()
        if on_done is not None:
            try:
                on_done()
            except Exception:  # noqa: BLE001
                LOGGER.debug("超时绑定清理失败", exc_info=True)

    _admit()
    try:
        future = _EXECUTOR.submit(invoke)
    except BaseException:
        _cleanup()  # 提交失败就把名额还回去，不然名额会慢性泄漏
        raise
    future.add_done_callback(_cleanup)

    try:
        return future.result(timeout=timeout_s)
    except TimeoutError as exc:
        # 3.11+ 里 concurrent.futures.TimeoutError 就是内置 TimeoutError，
        # 所以"等待超时"和"调用自己抛 timeout"必须靠 future.done() 区分：
        # done() 为真 = 异常来自调用内部（多半是客户端超时真的把连接掐了）。
        if future.done():
            metrics.incr("llm_timeout_total")
            metrics.incr("llm_client_timeout_fired_total")
            raise TimeoutError(f"LLM 调用超时（>{timeout_s:.0f}s）") from exc
        cancelled = future.cancel()  # 只对"还没开始跑"的任务有效，RUNNING 一律返回 False
        metrics.incr("llm_timeout_total")
        metrics.incr("llm_wait_timeout_total")
        if not cancelled:
            # 这条指标就是旧实现缺陷的量化：超时返回了，但线程还在跑。
            metrics.incr("llm_executor_stuck_total")
        raise TimeoutError(f"LLM 调用超时（>{timeout_s:.0f}s）") from exc
    except Exception as exc:
        if _is_timeout_exc(exc):
            metrics.incr("llm_timeout_total")
            metrics.incr("llm_client_timeout_fired_total")
            raise TimeoutError(f"LLM 调用超时（>{timeout_s:.0f}s）") from exc
        raise


def _invoke_messages(
    target: Any, messages: Sequence[Any], timeout_s: float, *, bind_client: bool = True
) -> Any:
    """真正的调用入口：先尝试让调用本身可中断，再上线程池兜底。"""
    on_done: Callable[[], None] | None = None
    mode = "executor"
    if bind_client:
        binding = bind_request_deadline(target, timeout_s)
        target, mode, on_done = binding.model, binding.mode, binding.close
    metrics.incr(f"llm_deadline_mode_{mode}_total")
    return _invoke_in_pool(lambda: target.invoke(messages), timeout_s, on_done=on_done)


def invoke_with_timeout(model: Any, messages: Sequence[Any], timeout_s: float = 90.0) -> Any:
    """给模型调用套一个"真"超时（默认 90s）。

    兼容性：签名与旧版完全一致（nodes.py / rerank.py 不需要改）。
    为什么超时做在调用边界而不是 get_model()：不同调用点的合理上限差很多
    （意图解析 30~45s、结果解读 120s），做在边界最省事；但边界只能改客户端，
    所以"给模型副本换客户端"这件事也放在这里做。
    """
    return _invoke_messages(model, messages, max(0.001, float(timeout_s)))


# ── 熔断器：三态 + 可选的 Redis 跨进程计数 ─────────────────────
_REDIS: Any = None
_REDIS_LOCK = threading.Lock()
_REDIS_RETRY_S = max(1.0, float(_env_num("OMICS_BREAKER_REDIS_RETRY_S", 5.0)))
# Redis 出错后的短暂冷却：不冷却的话，Redis 挂掉期间每一次 LLM 调用都要先撞一次连接超时，
# 熔断降级本身反而变成新的延迟来源。
_REDIS_DOWN_UNTIL = 0.0


def configure_breaker_backend(redis: Any = None) -> None:
    """模块级配置入口：把 Redis 客户端交给熔断层，失败计数从此跨进程共享。

    为什么是模块级而不是构造函数参数：nodes.py:57 的 LLM_BREAKER 在 import 期就建好了，
    web.py 后来才拿到 Redis，没有机会再传进去。所以 CircuitBreaker 在每次操作时
    读这个模块级后端 —— 调度方只需要在 web.py 启动后调一次：

        from omics_agent import resilience
        resilience.configure_breaker_backend(CACHE.redis)

    参数只要求鸭子类型（get/set/incr/expire/delete），redis.Redis 或 fakeredis 都行。
    传 None = 退回进程内计数。
    """
    global _REDIS, _REDIS_DOWN_UNTIL
    with _REDIS_LOCK:
        _REDIS = redis
        _REDIS_DOWN_UNTIL = 0.0
    metrics.incr("breaker_backend_configured_total")
    metrics.incr(
        "breaker_backend_redis_total" if redis is not None else "breaker_backend_local_total"
    )
    LOGGER.info("熔断器后端 = %s", "redis（跨进程）" if redis is not None else "local（进程内）")


# 别名：两种命名都顺手，接线的人不用记到底是 configure 还是 set
set_breaker_redis = configure_breaker_backend


def breaker_backend_mode() -> str:
    """当前熔断计数的实际后端：redis = 跨进程语义，local = 已降级到进程内。"""
    with _REDIS_LOCK:
        configured = _REDIS is not None
    if not configured:
        return "local"
    return "local" if time.monotonic() < _REDIS_DOWN_UNTIL else "redis"


def _redis_for_use() -> Any:
    with _REDIS_LOCK:
        client = _REDIS
    if client is None:
        return None
    if time.monotonic() < _REDIS_DOWN_UNTIL:  # 冷却期内直接走本地，不去撞一个可能已挂的 Redis
        return None
    return client


def _redis_failed(exc: BaseException) -> None:
    """Redis 后端出错：计指标 + 进入冷却，本次及后续调用自动退回进程内计数。"""
    global _REDIS_DOWN_UNTIL
    _REDIS_DOWN_UNTIL = time.monotonic() + _REDIS_RETRY_S
    metrics.incr("breaker_redis_error_total")
    metrics.incr("breaker_redis_fallback_total")
    LOGGER.warning("熔断器的 Redis 后端不可用，暂时退回进程内计数：%s", exc)


class CircuitBreaker:
    """三态熔断器：closed → open → half_open → closed。

    失败计数有两套后端，本地影子永远维护（Redis 掉线时立刻有一份可用的计数接管），
    Redis 可用时以 Redis 为准 —— 3 个 worker 看到的是同一份失败计数。
    这是本轮 P2 的修复：原实现只有进程内计数，等于把阈值乘以 worker 数。
    """

    def __init__(self, name: str = "llm", failure_threshold: int = 5, reset_timeout: float = 30.0) -> None:
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.reset_timeout = float(reset_timeout)
        self._lock = threading.Lock()
        self._state = "closed"
        self._failures = 0
        self._opened_at = 0.0

    # ── 状态读取（Redis 可用时以 Redis 为准）─────────────────
    @property
    def state(self) -> str:
        remote = self._remote_state()
        if remote is not None:
            return remote
        with self._lock:
            return self._state

    @property
    def failures(self) -> int:
        remote = self._remote_failures()
        if remote is not None:
            return remote
        with self._lock:
            return self._failures

    def stats(self) -> dict[str, Any]:
        """给健康检查用：这个闸门现在是什么状态、计数是哪套后端给的。"""
        return {
            "name": self.name,
            "state": self.state,
            "failures": self.failures,
            "failure_threshold": self.failure_threshold,
            "reset_timeout_s": self.reset_timeout,
            "backend": breaker_backend_mode(),
        }

    # ── 决策 ────────────────────────────────────────────
    def allow(self) -> bool:
        """是否放行这次调用；open 且冷却未到时返回 False。"""
        remote = self._remote_allow()
        if remote is not None:
            metrics.incr("breaker_decision_redis_total")
            return remote
        metrics.incr("breaker_decision_local_total")
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
        client = _redis_for_use()
        if client is None:
            return
        try:
            for part in ("fail", "open"):
                client.delete(self._key(part))
        except Exception as exc:  # noqa: BLE001
            _redis_failed(exc)

    def record_failure(self) -> None:
        # 1) 本地影子：无条件维护，Redis 掉线时它立刻升级为唯一计数
        with self._lock:
            self._failures += 1
            failures = self._failures
            opened_locally = False
            if self._state == "half_open" or failures >= self.failure_threshold:
                opened_locally = self._state != "open"
                self._state = "open"
                self._opened_at = time.monotonic()
        if opened_locally:
            self._count_open()

        # 2) Redis：INCR 得到跨进程计数，EXPIRE 让计数有个窗口，别永久累积
        client = _redis_for_use()
        if client is None:
            return
        try:
            fail_key, open_key = self._key("fail"), self._key("open")
            count = int(client.incr(fail_key))
            if count == 1:
                # 只在窗口起点设一次 TTL，避免每次失败都把窗口往后推（否则永远等不到清零）
                client.expire(fail_key, self._window_ttl())
            if count >= self.failure_threshold:
                # open 键带 EX = 冷却期；冷却到期键自然消失 = 半开，下一个探针放行。
                # value 里留时间戳和计数，redis-cli 一看就知道是谁、什么时候把闸门拉下来的。
                client.set(open_key, f"{time.time():.0f}:{count}", ex=self._cooldown_ttl())
                if count == self.failure_threshold:
                    # 只有"刚好跨过阈值"的那一次算跨进程熔断事件，否则每次失败都会 +1
                    metrics.incr("breaker_open_remote_total")
        except Exception as exc:  # noqa: BLE001
            _redis_failed(exc)

    # ── 内部小工具 ───────────────────────────────────────
    def _count_open(self) -> None:
        metrics.incr("breaker_open_total")
        metrics.incr(f"breaker_open_{self.name}_total")

    def _key(self, part: str) -> str:
        return f"omics:breaker:{self.name}:{part}"

    def _cooldown_ttl(self) -> int:
        return max(1, int(round(self.reset_timeout)) or 1)

    def _window_ttl(self) -> int:
        # 失败计数窗口要比冷却期长：冷却结束后第一个探针失败时，
        # 计数还在阈值之上 → 立刻重新打开（半开语义），不用再攒满 5 次。
        return max(60, self._cooldown_ttl() * 2)

    def _remote_allow(self) -> bool | None:
        client = _redis_for_use()
        if client is None:
            return None
        try:
            return client.get(self._key("open")) is None
        except Exception as exc:  # noqa: BLE001
            _redis_failed(exc)
            return None

    def _remote_state(self) -> str | None:
        client = _redis_for_use()
        if client is None:
            return None
        try:
            return "open" if client.get(self._key("open")) is not None else "closed"
        except Exception as exc:  # noqa: BLE001
            _redis_failed(exc)
            return None

    def _remote_failures(self) -> int | None:
        client = _redis_for_use()
        if client is None:
            return None
        try:
            raw = client.get(self._key("fail"))
            return int(raw or 0)
        except Exception as exc:  # noqa: BLE001
            _redis_failed(exc)
            return None


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
        except PoolSaturatedError as exc:
            # 本地容量饱和 ≠ 网关故障：不记熔断失败（否则本地拥塞会把闸门自己拉下来），
            # 但退避重试 —— 给在途调用一点时间把名额还回来。
            last_exc = exc
            metrics.incr(f"{label}_capacity_total")
            if i >= attempts - 1:
                break
            metrics.incr(f"{label}_retry_total")
            sleep(max(0.0, min(max_delay, base_delay * (2 ** i))))
            continue
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
    timeout_s = max(0.001, float(timeout_s))

    # 超时绑在基座模型上：with_structured_output 返回的是 Runnable 包装，
    # 它自己身上没有可改的 HTTP 客户端，绑在包装上等于没绑。
    binding = bind_request_deadline(model, timeout_s)
    base_model = binding.model
    metrics.incr(f"llm_deadline_mode_{binding.mode}_total")
    try:
        for attempt in range(max(1, int(attempts))):
            if breaker is not None and not breaker.allow():
                last_exc = CircuitOpenError("llm_structured: 熔断中，跳过调用")
                metrics.incr("llm_structured_short_circuit_total")
                break
            try:
                structured = base_model.with_structured_output(schema)
                # bind_client=False：客户端超时已经在上面绑过了，别对包装对象再绑一次
                out = _invoke_messages(structured, msgs, timeout_s, bind_client=False)
            except PoolSaturatedError as exc:
                # 名额不够属于本地拥塞：不算网关失败，但值得重排一次
                last_exc = exc
                metrics.incr("llm_structured_capacity_total")
                if attempt >= attempts - 1:
                    break
                metrics.incr("llm_structured_retry_total")
                sleep(min(4.0, 0.5 * (2 ** attempt)))
                continue
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
    finally:
        binding.close()

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
    ("clustering", ("聚类", "分群", "cluster", "kmeans", "k-means", "分组模式", "亚型", "几群")),
    ("qa", ("哪些基因", "什么功能", "功能是", "参与", "通路", "注释", "介绍一下",
            "知识库", "相关基因", "什么是", "有关系")),
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


def resilience_stats() -> dict[str, Any]:
    """韧性层的整体状态：给健康检查/演示看，不用去翻私有变量。"""
    return {
        "pool": pool_stats(),
        "breaker_backend": breaker_backend_mode(),
        "counters": {
            "llm_timeout_total": metrics.METRICS.counter("llm_timeout_total"),
            "llm_executor_stuck_total": metrics.METRICS.counter("llm_executor_stuck_total"),
            "llm_pool_saturated_total": metrics.METRICS.counter("llm_pool_saturated_total"),
            "breaker_open_total": metrics.METRICS.counter("breaker_open_total"),
            "breaker_open_remote_total": metrics.METRICS.counter("breaker_open_remote_total"),
            "breaker_redis_fallback_total": metrics.METRICS.counter("breaker_redis_fallback_total"),
        },
    }
