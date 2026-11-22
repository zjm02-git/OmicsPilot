"""Redis Stream 任务队列：XADD → XREADGROUP → XACK → XAUTOCLAIM → 死信。

为什么换掉线程池（面试讲这个）：
  线程池是进程内的：进程一挂，排队中 + 执行中的任务全丢；
  Redis Stream 把队列和任务状态都放 Redis，多个 worker 进程共享同一消费组，
  一个 worker 崩了，它 PEL 里的未确认任务会被别的 worker 用 XAUTOCLAIM 接走。

四个关键机制（面试必讲）：
  1. at-least-once：XACK 之前任务不算完成，任何崩溃都会重投 → 处理必须幂等（job_id 去重）；
  2. PEL（Pending Entries List）：每条「读过但没 XACK」的消息都挂在上面，
     是「任务丢了能找回」的依据；
  3. 可见性超时：XAUTOCLAIM 的 min_idle_time，超过 N 秒没确认就认为 worker 死了，
     等价于 SQS 的 visibility timeout；
  4. 死信：重试超上限的消息进独立 stream，防止毒消息卡死消费循环。
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from typing import Any

from redis.exceptions import ResponseError

from omics_agent import metrics

STREAM = "omics:jobs"
GROUP = "omics:workers"
DEAD_STREAM = "omics:jobs:dead"
JOB_KEY_PREFIX = "omics:job:"


class StreamQueue:
    """Redis Stream 队列薄封装：只管消息进出、超时回收、死信，不管业务。"""

    def __init__(
        self,
        redis_client: Any,
        stream: str = STREAM,
        group: str = GROUP,
        dead_stream: str = DEAD_STREAM,
        visible_timeout_s: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.redis = redis_client
        self.stream = stream
        self.group = group
        self.dead_stream = dead_stream
        self.visible_timeout_ms = max(1000, int(visible_timeout_s * 1000))
        self.max_retries = max(0, int(max_retries))

    def ensure_group(self) -> None:
        """幂等建消费组（已存在会抛 BUSYGROUP，忽略）。"""
        try:
            self.redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except Exception:  # noqa: BLE001 - 组已存在
            pass

    def enqueue(self, job_id: str, task_type: str, payload: dict) -> str:
        """投递任务：消息体带 job_id，worker 侧靠它做幂等去重。"""
        return self.redis.xadd(
            self.stream,
            {"job_id": job_id, "type": task_type, "payload": json.dumps(payload), "retry": 0},
        )

    def new_consumer(self) -> str:
        """消费者名：hostname + pid + 随机后缀，保证每个 worker 唯一。"""
        return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"

    def read(self, consumer: str, count: int = 1, block_ms: int = 5000) -> list[tuple[str, dict]]:
        """阻塞读新消息（只读 '>' 之后，不碰自己 PEL 里没确认的）。"""
        try:
            resp = self.redis.xreadgroup(
                self.group, consumer, {self.stream: ">"}, count=count, block=block_ms
            )
        except ResponseError:
            # stream/group 不存在（Redis 被 flush 等）：重建后返回空，下一轮恢复正常阻塞读
            self.ensure_group()
            return []
        out: list[tuple[str, dict]] = []
        for _stream, entries in resp:
            for msg_id, data in entries:
                out.append((msg_id, data))
        return out

    def ack(self, msg_id: str) -> None:
        self.redis.xack(self.stream, self.group, msg_id)

    def claim_stale(self, consumer: str, count: int = 100) -> list[tuple[str, dict]]:
        """XAUTOCLAIM：把「超过可见性超时还没确认」的消息抢到自己名下处理。"""
        try:
            _next_start, claimed, _deleted = self.redis.xautoclaim(
                self.stream, self.group, consumer,
                min_idle_time=self.visible_timeout_ms, start_id="0-0", count=count,
            )
        except ResponseError:
            self.ensure_group()
            return []
        if claimed:
            metrics.incr("queue_claim_total", value=len(claimed))
        return claimed

    def retry(self, msg_id: str, data: dict, retry: int) -> None:
        """重投：失败但没超上限，消息重新进队尾（retry+1），确认旧消息。"""
        self.redis.xadd(
            self.stream,
            {"job_id": data["job_id"], "type": data["type"], "payload": data["payload"], "retry": retry},
        )
        self.redis.xack(self.stream, self.group, msg_id)
        metrics.incr("queue_retry_total")

    def dead_letter(self, msg_id: str, data: dict, error: str) -> None:
        """进死信：失败超上限，消息进独立 stream 留证据，确认原消息。"""
        self.redis.xadd(self.dead_stream, {**data, "error": error})
        self.redis.xack(self.stream, self.group, msg_id)
        metrics.incr("queue_dead_letter_total")

    def stats(self) -> dict[str, Any]:
        """队列健康度：待消费 / PEL 未确认 / 死信。"""
        try:
            pending = self.redis.xpending(self.stream, self.group)
            pending_count = pending.get("pending", 0) if isinstance(pending, dict) else len(pending)
        except Exception:  # noqa: BLE001 - 组还没建时 xpending 报错
            pending_count = 0
        return {
            "queue_len": int(self.redis.xlen(self.stream) or 0),
            "pending": pending_count,
            "dead_len": int(self.redis.xlen(self.dead_stream) or 0),
        }
