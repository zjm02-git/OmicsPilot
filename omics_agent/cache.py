"""缓存层：Redis 主存 + 进程内 LRU 兜底，TTL 抖动、并发击穿合并、数据集指纹入 key。

修掉上一版的三个真问题：
  1. key 只哈希了 user_request → 换一份上传数据问同一句话，会拿到上一份报告（正确性 bug）；
  2. Redis 一挂 /analyze 直接 500（可用性 bug）→ 现在降级到进程内缓存，并打降级指标；
  3. 同一句话并发进来时 N 个请求各跑一遍 25s 的图（重复计算）→
     用 Redis SET NX 抢"计算权"，其余请求等待并复用同一份结果（single-flight）。

TTL 抖动：给过期时间加 ±10% 随机量，避免同一批报告在同一秒集体过期
（缓存雪崩的经典成因：整点批量写入 + 固定 TTL）。
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

from omics_agent import metrics
from omics_agent.dataio import dataset_fingerprint

DEFAULT_TTL = 24 * 3600


class TTLCache:
    """带 TTL 的 LRU：Redis 不可用时的本地兜底，也用于短生命周期对象。"""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max(1, int(max_entries))
        self._data: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expire_at = item
            if expire_at <= time.monotonic():
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: float = DEFAULT_TTL) -> None:
        with self._lock:
            self._data[key] = (value, time.monotonic() + max(1.0, float(ttl)))
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class ReportCache:
    """报告缓存：Redis 优先，挂了自动降级到进程内；带 single-flight 与 TTL 抖动。"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        ttl: float = DEFAULT_TTL,
        jitter: float = 0.10,
        fallback: TTLCache | None = None,
        redis_client: Any = None,
        connect: bool = True,
    ) -> None:
        self.ttl = float(ttl)
        self.jitter = max(0.0, min(0.5, float(jitter)))
        self._fallback = fallback or TTLCache(max_entries=256)
        self._lock = threading.Lock()
        self._local_locks: set[str] = set()
        self.redis: Any = None

        if redis_client is not None:
            self.redis = redis_client
        elif connect:
            try:
                import redis  # 局部导入：测试/离线环境不装也能用本地兜底

                client = redis.Redis(
                    host=host or os.getenv("REDIS_HOST", "127.0.0.1"),
                    port=int(port or os.getenv("REDIS_PORT", "6379")),
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0,
                    decode_responses=True,
                )
                client.ping()
                self.redis = client
            except Exception:  # noqa: BLE001 - 连不上就是降级，不该让服务起不来
                self.redis = None
                metrics.incr("cache_redis_unavailable_total")

    # ── 基础读写 ──────────────────────────────────────────
    @property
    def redis_available(self) -> bool:
        return self.redis is not None

    def ping(self) -> bool:
        if self.redis is None:
            return False
        try:
            return bool(self.redis.ping())
        except Exception:  # noqa: BLE001
            return False

    def get(self, key: str) -> Any:
        if self.redis is not None:
            try:
                return self.redis.get(key)
            except Exception:  # noqa: BLE001 - Redis 挂了：降级读本地，并记账
                metrics.incr("cache_redis_error_total")
                metrics.incr("cache_fallback_read_total")
                return self._fallback.get(key)
        return self._fallback.get(key)

    def get_source(self, key: str) -> tuple[Any, str]:
        """返回 (值, 来源)。来源用于在响应里区分 "redis" / "local" / "miss"。"""
        if self.redis is not None:
            try:
                value = self.redis.get(key)
                return (value, "redis") if value is not None else (None, "miss")
            except Exception:  # noqa: BLE001
                metrics.incr("cache_redis_error_total")
                metrics.incr("cache_fallback_read_total")
                value = self._fallback.get(key)
                return (value, "local") if value is not None else (None, "miss")
        value = self._fallback.get(key)
        return (value, "local") if value is not None else (None, "miss")

    def set(self, key: str, value: Any, ttl: float | None = None) -> bool:
        ttl_eff = self.ttl if ttl is None else float(ttl)
        ttl_eff *= 1.0 + random.uniform(-self.jitter, self.jitter)  # 抖动：防批量同时过期
        ttl_eff = max(1.0, ttl_eff)

        wrote_redis = False
        if self.redis is not None:
            try:
                self.redis.set(key, value, ex=max(1, int(ttl_eff)))
                wrote_redis = True
            except Exception:  # noqa: BLE001
                metrics.incr("cache_redis_error_total")
                metrics.incr("cache_fallback_write_total")
        # 双写本地：Redis 抖动的那几秒，本地仍能命中，不至于把流量全打到图执行上
        self._fallback.set(key, value, ttl=ttl_eff)
        return wrote_redis

    def delete(self, key: str) -> None:
        if self.redis is not None:
            try:
                self.redis.delete(key)
            except Exception:  # noqa: BLE001
                metrics.incr("cache_redis_error_total")
        self._fallback.delete(key)

    # ── single-flight ─────────────────────────────────────
    @contextmanager
    def inflight(self, key: str, ttl: int = 120):
        """抢"计算权"：yield True = 由本请求负责计算；False = 别人正在算，去等结果。"""
        mode = self._acquire(key, ttl)
        try:
            yield mode is not None
        finally:
            self._release(key, mode)

    def _acquire(self, key: str, ttl: int) -> str | None:
        if self.redis is not None:
            try:
                ok = bool(self.redis.set(f"lock:{key}", "1", nx=True, ex=max(5, int(ttl))))
                if ok:
                    return "redis"
                return None
            except Exception:  # noqa: BLE001 - Redis 挂了退回本地锁
                metrics.incr("cache_redis_error_total")
        with self._lock:
            if key in self._local_locks:
                return None
            self._local_locks.add(key)
            return "local"

    def _release(self, key: str, mode: str | None) -> None:
        if mode == "redis":
            try:
                self.redis.delete(f"lock:{key}")
            except Exception:  # noqa: BLE001
                metrics.incr("cache_redis_error_total")
            return
        if mode == "local":
            with self._lock:
                self._local_locks.discard(key)

    def wait_for(self, key: str, timeout: float = 90.0, interval: float = 0.05) -> Any:
        """等别的请求把结果写进缓存；超时返回 None（调用方自己算，宁可重复也不挂死）。"""
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            value = self.get(key)
            if value is not None:
                return value
            time.sleep(interval)
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "redis_available": self.redis_available,
            "local_entries": len(self._fallback),
            "ttl_s": int(self.ttl),
            "ttl_jitter": self.jitter,
        }


def build_key(user_request: str, dataset_path: str, version: str = "v3") -> str:
    """缓存 key = 版本 + 归一化问题 + 数据集绝对路径 + 数据集指纹。

    指纹（文件名+字节数+mtime）保证"换数据不串报告"；路径入 key 允许同一份
    数据被上传到两个目录时各自独立缓存，避免跨任务误命中。
    """
    fingerprint = dataset_fingerprint(dataset_path)
    normalized = " ".join((user_request or "").split())
    raw = f"{version}|{normalized}|{os.path.abspath(dataset_path)}|{fingerprint}"
    return "report:" + version + ":" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
