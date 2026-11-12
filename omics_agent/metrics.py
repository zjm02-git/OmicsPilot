"""可观测层：进程内指标（零第三方依赖，不新增任何安装项）。

为什么自己写：requirements 里没有 prometheus_client，而下面三件事必须有答案——
  1. 接口到底慢在哪：按路由统计 P50/P90/P95/P99（分位数比平均值有用得多）；
  2. 缓存到底有没有用：命中、击穿合并、Redis 故障降级各自的次数；
  3. 这条日志属于哪次请求：X-Request-ID 用 contextvar 贯穿到最深的函数里。

取舍：样本环形截断到 2000 条，内存恒定；进程内聚合，多 worker 各自暴露。
生产环境迁移点：把 snapshot() 的计数器和分位数组换成 Prometheus 的
Counter/Histogram，路由标签用 request.scope["route"].path，语义一致。
"""

from __future__ import annotations

import contextvars
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterable

MAX_SAMPLES = 2000

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def new_request_id() -> str:
    """给一次请求/一次任务生成短 ID：日志和响应头里都能对上。"""
    return uuid.uuid4().hex[:12]


def set_request_id(value: str) -> contextvars.Token:
    return _request_id.set(value or "-")


def get_request_id() -> str:
    return _request_id.get()


def reset_request_id(token: contextvars.Token) -> None:
    try:
        _request_id.reset(token)
    except (ValueError, LookupError):  # 跨线程 reset 会失败，忽略即可
        pass


class Metrics:
    """计数器 + 计时器（环形采样分位数）。所有方法线程安全。"""

    def __init__(self, max_samples: int = MAX_SAMPLES) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._samples: dict[str, list[float]] = {}
        self._max_samples = max(16, int(max_samples))
        self._started = time.time()

    # ── 写 ────────────────────────────────────────────────
    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + float(value)

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            samples = self._samples.setdefault(name, [])
            if len(samples) >= self._max_samples:
                del samples[0]
            samples.append(float(ms))

    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000.0)

    # ── 读 ────────────────────────────────────────────────
    def counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def percentiles(self, name: str, qs: Iterable[float] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, float]:
        with self._lock:
            samples = sorted(self._samples.get(name, []))
        if not samples:
            return {}
        n = len(samples)
        out: dict[str, float] = {}
        for q in qs:
            idx = min(n - 1, max(0, int(round(q * (n - 1)))))
            out[f"p{int(q * 100)}_ms"] = round(samples[idx], 2)
        out["max_ms"] = round(samples[-1], 2)
        out["mean_ms"] = round(sum(samples) / n, 2)
        out["count"] = n
        return out

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = {k: round(v, 3) for k, v in self._counters.items()}
            samples = {k: sorted(v) for k, v in self._samples.items() if v}

        timers: dict[str, dict[str, float]] = {}
        for name, ordered in samples.items():
            n = len(ordered)
            entry: dict[str, float] = {
                "count": n,
                "mean_ms": round(sum(ordered) / n, 2),
                "max_ms": round(ordered[-1], 2),
            }
            for q in (0.5, 0.9, 0.95, 0.99):
                idx = min(n - 1, max(0, int(round(q * (n - 1)))))
                entry[f"p{int(q * 100)}_ms"] = round(ordered[idx], 2)
            timers[name] = entry

        return {
            "request_id": get_request_id(),
            "uptime_s": round(time.time() - self._started, 1),
            "counters": counters,
            "timers": timers,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._samples.clear()


# 进程内单例：所有模块 `from omics_agent import metrics` 后直接用函数接口
METRICS = Metrics()


def incr(name: str, value: float = 1.0) -> None:
    METRICS.incr(name, value)


def observe_ms(name: str, ms: float) -> None:
    METRICS.observe(name, ms)


def timer(name: str):
    return METRICS.timer(name)


def snapshot() -> dict[str, Any]:
    return METRICS.snapshot()
