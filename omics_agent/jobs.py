"""异步任务执行器：把 20~30 秒的图执行从 HTTP 请求里挪出去。

动机（实测）：一次差异分析 25 秒，其中 LLM 解读占 ~18 秒。同步接口下前端只能挂着长连接等，
网关/浏览器一超时就是"算完了没人收"；同时几个用户进来，uvicorn 的工作线程会被
这几条长请求占满，健康检查也会跟着变慢。用户要的其实是"提交 → 拿 job_id → 轮询"。

设计取舍：
  - 线程池而非 Celery/RabbitMQ：项目体量还没有跨进程/跨机需求，线程池 + 注册表可维护；
    真要上队列，把 submit/get 换成 broker 调用即可（接口不变）。
  - 任务状态放进程内，重启即失效；因此每个任务同时把结果写 DB 历史（/history 可查）。
"""

from __future__ import annotations

import contextvars
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from omics_agent import metrics

# 当前任务（在 worker 线程里设置）：让深层代码也能报告"我到哪一步了"
_CURRENT_JOB: contextvars.ContextVar[dict | None] = contextvars.ContextVar("omics_job", default=None)


def progress(stage: str) -> None:
    """由 worker 内的代码调用：更新当前任务的阶段文案（没在任务里就是空操作）。"""
    job = _CURRENT_JOB.get()
    if job is not None:
        job["stage"] = str(stage)
        job["updated_at"] = time.time()


class JobRegistry:
    """任务注册表 + 有界线程池。"""

    def __init__(self, max_workers: int = 4, ttl: float = 3600.0, max_jobs: int = 200) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="omics-job")
        self._jobs: "dict[str, dict[str, Any]]" = {}
        self._lock = threading.Lock()
        self.ttl = float(ttl)
        self.max_jobs = max(8, int(max_jobs))

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "排队中",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "elapsed_s": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._evict_locked()
        metrics.incr("job_submitted_total")
        self._executor.submit(self._run, job, fn, args, kwargs)
        return job_id

    def _run(self, job: dict[str, Any], fn: Callable[..., Any], args: tuple, kwargs: dict) -> None:
        token = _CURRENT_JOB.set(job)
        job["status"] = "running"
        job["started_at"] = time.time()
        job["stage"] = "执行中"
        try:
            with metrics.timer("job.run_ms"):
                job["result"] = fn(*args, **kwargs)
            job["status"] = "done"
            metrics.incr("job_done_total")
        except Exception as exc:  # noqa: BLE001 - 任务失败要落到任务状态里，不能只打日志
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            metrics.incr("job_error_total")
        finally:
            job["finished_at"] = time.time()
            if job["started_at"]:
                job["elapsed_s"] = round(job["finished_at"] - job["started_at"], 2)
            _CURRENT_JOB.reset(token)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def _evict_locked(self) -> None:
        now = time.time()
        stale = [
            jid for jid, job in self._jobs.items()
            if job["status"] in ("done", "error")
            and job.get("finished_at") and now - job["finished_at"] > self.ttl
        ]
        for jid in stale:
            self._jobs.pop(jid, None)
        while len(self._jobs) > self.max_jobs:
            for jid in list(self._jobs):
                if self._jobs[jid]["status"] in ("done", "error"):
                    self._jobs.pop(jid, None)
                    break
            else:
                break

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job["status"]] = counts.get(job["status"], 0) + 1
        return {"jobs_total": len(self._jobs), "by_status": counts}
