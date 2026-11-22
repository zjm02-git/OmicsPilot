"""异步任务队列：Redis Stream 真队列（XADD → XREADGROUP → XACK → XAUTOCLAIM → 死信）。

动机（为什么换掉线程池）：
  线程池是进程内的：进程一挂，排队中 + 执行中的任务全丢；
  Redis Stream 把队列和任务状态都放 Redis，多个 worker 进程共享同一消费组，
  一个 worker 崩了，它 PEL 里的未确认任务会被别的 worker 用 XAUTOCLAIM 接走。
  （kill -9 掉一个 worker，任务被存活 worker 接走 —— 这是"真队列"才有的证据。）

四个必须讲清的点：
  1. at-least-once：XACK 之前任务不算完成，崩溃会重投 → 处理必须幂等（job_id 去重）；
  2. PEL：每条「读过没 XACK」的消息都挂在 PEL 上，是「任务能找回」的依据；
  3. 可见性超时：XAUTOCLAIM 的 min_idle_time，超过 N 秒没确认 = worker 死了，等价 SQS visibility timeout；
  4. 死信：重试超上限进独立 stream，防毒消息卡死消费循环。
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from typing import Any, Callable

from omics_agent import metrics
from omics_agent.queue import DEAD_STREAM, GROUP, JOB_KEY_PREFIX, STREAM, StreamQueue

# 当前任务上下文（worker 执行时设置）：progress() 靠它知道往哪个 Redis Hash 写阶段
_CURRENT_JOB: contextvars.ContextVar[dict | None] = contextvars.ContextVar("omics_job", default=None)


def progress(stage: str) -> None:
    """由任务代码调用：把当前任务的阶段写进 Redis（没在任务里就是空操作）。

    接口不变：web.py 的 _run_graph 里 jobs_mod.progress(...) 不用改。
    """
    ctx = _CURRENT_JOB.get()
    if ctx is None:
        return
    try:
        ctx["redis"].hset(
            JOB_KEY_PREFIX + ctx["job_id"],
            mapping={"stage": str(stage), "updated_at": time.time()},
        )
    except Exception:  # noqa: BLE001 - 阶段更新失败不影响主流程
        pass


class JobRegistry:
    """任务注册表：状态存 Redis Hash，队列走 Redis Stream，worker 消费执行。"""

    def __init__(
        self,
        redis_client: Any,
        ttl: float = 3600.0,
        visible_timeout_s: float = 60.0,
        max_retries: int = 3,
        stream: str = STREAM,
        group: str = GROUP,
        dead_stream: str = DEAD_STREAM,
    ) -> None:
        self.redis = redis_client
        self.queue = StreamQueue(
            redis_client,
            stream=stream,
            group=group,
            dead_stream=dead_stream,
            visible_timeout_s=visible_timeout_s,
            max_retries=max_retries,
        )
        self.ttl = max(60, int(ttl))
        # 心跳间隔取可见性超时的 1/3：连丢 2 次心跳才可能被误判为"worker 已死"
        self.heartbeat_interval_s = max(2.0, self.queue.visible_timeout_ms / 1000.0 / 3.0)
        self._consumer_name = "unknown"
        self._handlers: dict[str, Callable] = {}

    # ── 提交侧（web 用）────────────────────────────
    def register(self, task_type: str, fn: Callable) -> None:
        """注册「任务类型 → 处理函数」。幂等：重复注册覆盖。"""
        self._handlers[task_type] = fn

    def submit(self, task_type: str, payload: dict, owner: Any = None) -> str:
        """提交任务：写状态（queued）+ 入队（XADD）。返回 job_id。

        owner 是任务归属（user_id）：/jobs/{id} 靠它做资源级鉴权，
        不写的话任何登录用户拿到 job_id 就能读别人的完整报告。
        """
        if self.redis is None:
            raise RuntimeError("任务队列依赖 Redis，但 Redis 当前不可用")
        job_id = uuid.uuid4().hex[:12]
        fields = {
            "job_id": job_id,
            "type": task_type,
            "status": "queued",
            "stage": "排队中",
            "created_at": time.time(),
            "retry": 0,
        }
        if owner is not None:
            fields["user_id"] = str(owner)
        self._set_job(job_id, fields)
        self.queue.enqueue(job_id, task_type, payload)
        metrics.incr("job_submitted_total")
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        if self.redis is None:
            return None
        try:
            raw = self.redis.hgetall(JOB_KEY_PREFIX + job_id)
        except Exception:  # noqa: BLE001 - Redis 中途掉线：查任务失败不等于 500
            return None
        if not raw:
            return None
        job = dict(raw)
        # result 存的是 JSON 字符串，反序列化回对象，保持旧接口（result 是 dict/int）
        if job.get("result"):
            try:
                job["result"] = json.loads(job["result"])
            except Exception:  # noqa: BLE001
                pass
        for f in ("retry",):
            if f in job:
                try:
                    job[f] = int(job[f])
                except (TypeError, ValueError):
                    pass
        return job

    def stats(self) -> dict[str, Any]:
        if self.redis is None:
            return {"jobs_total": 0, "by_status": {}, "queue": self.queue.stats(),
                    "available": False}
        counts: dict[str, int] = {}
        try:
            for key in self.redis.scan_iter(match=JOB_KEY_PREFIX + "*", count=100):
                raw = self.redis.hgetall(key)
                if raw:
                    st = raw.get("status", "unknown")
                    counts[st] = counts.get(st, 0) + 1
        except Exception:  # noqa: BLE001 - 探活接口不能因为依赖挂了就 500
            return {"jobs_total": 0, "by_status": counts, "queue": self.queue.stats(),
                    "available": False}
        return {
            "jobs_total": sum(counts.values()),
            "by_status": counts,
            "queue": self.queue.stats(),
            "available": True,
        }

    # ── 内部：状态读写 ─────────────────────────────
    def _set_job(self, job_id: str, fields: dict) -> None:
        mapping = {k: ("" if v is None else str(v)) for k, v in fields.items()}
        key = JOB_KEY_PREFIX + job_id
        self.redis.hset(key, mapping=mapping)
        self.redis.expire(key, self.ttl)

    # ── worker 消费侧 ─────────────────────────────
    def run_worker(self, consumer: str | None = None, block_ms: int = 2000) -> None:
        """阻塞消费循环：先 XAUTOCLAIM 接走超时任务，再 XREADGROUP 读新任务。"""
        self.queue.ensure_group()
        consumer = consumer or self.queue.new_consumer()
        self._consumer_name = consumer
        print(
            f"[worker] 启动 consumer={consumer}，可见性超时={self.queue.visible_timeout_ms}ms，"
            f"心跳间隔={self.heartbeat_interval_s}s",
            flush=True,
        )
        while True:
            # 1) 接管别人挂了没确认的任务（PEL 里超时的）
            for msg_id, data in self.queue.claim_stale(consumer):
                self._handle(msg_id, data)
            # 2) 读新任务
            for msg_id, data in self.queue.read(consumer, count=1, block_ms=block_ms):
                self._handle(msg_id, data)

    def _handle(self, msg_id: str, data: dict) -> None:
        job_id = data["job_id"]
        retry = int(data.get("retry", 0))
        # 幂等：已经完成/失败的任务直接确认，不重复执行（at-least-once 的必要配套）
        existing = self.redis.hgetall(JOB_KEY_PREFIX + job_id)
        if existing and existing.get("status") in ("done", "error"):
            self.queue.ack(msg_id)
            metrics.incr("job_dedup_total")
            return

        # 「PEL 里超时」只说明原 worker 没 XACK，不等于它死了：
        # 单步 LLM 超时可能比可见性超时更长，光靠 XAUTOCLAIM 会把还在跑的任务抢走跑两遍
        # （双倍 token、双份历史）。所以先看心跳，活着就不接。
        if existing and existing.get("status") == "running":
            try:
                beat = float(existing.get("heartbeat_at") or 0.0)
            except (TypeError, ValueError):
                beat = 0.0
            if time.time() - beat < self.queue.visible_timeout_ms / 1000.0:
                # 注意：这里**不能 XACK**。确认掉就意味着"这条任务处理完了"，
                # 万一原 worker 其实已经死了（心跳恰好在窗口内），任务就永久丢了。
                # 留在 PEL 里即可：等它真死了，心跳过期后下一轮就会被正常接管。
                metrics.incr("job_claim_rejected_total")
                return

        started = time.time()
        self._set_job(job_id, {
            "status": "running",
            "stage": "执行中",
            "started_at": started,
            "heartbeat_at": started,
            "worker": self._consumer_name,
        })
        token = _CURRENT_JOB.set({"job_id": job_id, "redis": self.redis})
        stop_beat = threading.Event()
        beat_thread = threading.Thread(
            target=self._heartbeat, args=(job_id, stop_beat), daemon=True
        )
        beat_thread.start()
        try:
            result = self._dispatch(data["type"], json.loads(data["payload"]))
            # 不覆盖 stage：保留 progress() 最后上报的阶段文案
            self._set_job(job_id, {
                "status": "done",
                "result": json.dumps(result, ensure_ascii=False),
                "finished_at": time.time(),
                "elapsed_s": round(time.time() - started, 2),
            })
            self.queue.ack(msg_id)
            metrics.incr("job_done_total")
        except Exception as exc:  # noqa: BLE001 - 失败要落状态 + 决定重试/死信
            error = f"{type(exc).__name__}: {exc}"
            if retry >= self.queue.max_retries:
                self.queue.dead_letter(msg_id, data, error)
                self._set_job(job_id, {
                    "status": "error", "stage": "失败(进死信)", "error": error,
                    "finished_at": time.time(), "elapsed_s": round(time.time() - started, 2),
                })
                metrics.incr("job_dead_total")
            else:
                self.queue.retry(msg_id, data, retry + 1)
                self._set_job(job_id, {"status": "queued", "stage": "重试排队中", "retry": retry + 1})
                metrics.incr("job_retry_total")
        finally:
            stop_beat.set()
            _CURRENT_JOB.reset(token)

    def _heartbeat(self, job_id: str, stop: threading.Event) -> None:
        """执行期间定期续命：让别的 worker 知道这个任务还活着，别接管。"""
        while not stop.wait(self.heartbeat_interval_s):
            try:
                self.redis.hset(JOB_KEY_PREFIX + job_id, mapping={"heartbeat_at": time.time()})
            except Exception:  # noqa: BLE001 - 心跳失败不中断业务，最坏是被接管
                return

    def _dispatch(self, task_type: str, payload: dict) -> Any:
        handler = self._handlers.get(task_type)
        if handler is None:
            raise ValueError(f"未知任务类型：{task_type}")
        return handler(payload)
