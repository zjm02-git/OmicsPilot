"""Redis Stream 容错演示：kill -9 一个 worker，任务被存活 worker 接走。

这条跑通 = 「真队列」才有的容错能力：线程池做不到（进程一挂任务全丢），
而这里用裸 Redis Stream 手写出了 Celery 的核心机制，面试现场能直接复现。

用法（在容器里跑，Redis 在容器内、REDIS_HOST=redis 已配好）：
    docker compose exec worker python tools/demo_failover.py

可调参数（环境变量）：
    DEMO_VISIBLE_TIMEOUT_S   可见性超时（默认 5s，压小演示快）
    DEMO_SLEEP_S             慢任务 sleep 时长（默认 15s，必须 > 可见性超时）
"""
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis

from omics_agent import jobs as jm


def main() -> int:
    manual = "--manual" in sys.argv  # 手动模式：脚本不自动 kill，打印 pid 等你亲手 kill -9
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "127.0.0.1"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        decode_responses=True,
    )
    # 独立 stream/group，不碰生产队列（omics:jobs）
    S, G, D = "demo:failover:jobs", "demo:failover:group", "demo:failover:dead"
    visible_timeout_s = float(os.getenv("DEMO_VISIBLE_TIMEOUT_S", "5"))
    sleep_s = float(os.getenv("DEMO_SLEEP_S", "15"))

    reg = jm.JobRegistry(
        redis_client=r, ttl=180, visible_timeout_s=visible_timeout_s, max_retries=3,
        stream=S, group=G, dead_stream=D,
    )
    reg.register("slow", lambda p: (time.sleep(float(p.get("sleep", sleep_s))), "ok")[1])

    def worker() -> None:
        reg.run_worker(block_ms=1000)

    procs = [mp.Process(target=worker, daemon=True) for _ in range(2)]
    for p in procs:
        p.start()
    time.sleep(1.5)

    try:
        job_id = reg.submit("slow", {"sleep": sleep_s})
        print(f"[1] 提交任务 job_id={job_id}（慢任务 sleep={sleep_s}s）", flush=True)

        # 等某个 worker 领走任务、开始执行
        for _ in range(100):
            job = reg.get(job_id)
            if job and job["status"] == "running":
                break
            time.sleep(0.1)
        print(f"[2] 任务状态 = {job['status']}（已被某个 worker 领走）", flush=True)

        # 查 PEL：这条「读过没 XACK」的消息挂在哪个 consumer 名下 = 谁在跑它
        pend = r.xpending(S, G)
        print(f"[3] PEL（pending 列表）: {pend}", flush=True)
        consumer = None
        if isinstance(pend, dict):
            for c in (pend.get("consumers") or []):
                if c.get("pending", 0) > 0:
                    consumer = c["name"]
                    break
        if consumer is None:
            print("[!] 没找到正在执行的 consumer，退出", flush=True)
            return 1

        # consumer 名格式 hostname-pid-uuid，解析出执行 worker 的 pid，kill -9
        wpid = int(consumer.split("-")[1])
        if manual:
            print(
                f"[4] 执行中的 worker pid={wpid}。请你亲手 kill -9 它（另开一个终端执行）：\n"
                f"    docker compose exec worker sh -c 'kill -9 {wpid}'\n"
                f"    这里会一直等，等你 kill 完、任务被存活 worker 接走为止…",
                flush=True,
            )
        else:
            os.kill(wpid, 9)
            print(f"[4] kill -9 执行中的 worker（pid={wpid}），模拟进程崩溃", flush=True)

        # 等存活 worker 用 XAUTOCLAIM 接走（消息 idle 超过 visible_timeout 后）
        print(f"[5] 等 {visible_timeout_s}s 可见性超时，存活 worker 用 XAUTOCLAIM 接走…", flush=True)
        final = None
        t0 = time.time()
        while time.time() - t0 < sleep_s + 30:
            job = reg.get(job_id)
            if job and job["status"] == "done":
                final = job
                break
            time.sleep(0.2)
        if final is None:
            print("[!] 超时：任务没有被接走", flush=True)
            return 1

        print(
            f"[6] 任务最终 status={final['status']}，elapsed_s={final.get('elapsed_s')} "
            "（存活 worker 重新执行完成）",
            flush=True,
        )
        print("结论：kill -9 掉 worker，任务没丢，被存活 worker 接走 —— at-least-once 的实证", flush=True)
        return 0
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        r.delete(S, D)
        print("[cleanup] 已停 worker、清理 demo stream", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
