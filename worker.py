"""独立 worker 进程：消费 Redis Stream 里的任务。

用法（项目根目录）：
    python worker.py            # 起 1 个 worker（串行消费任务）
多起几个进程 = 多个消费者，共享消费组，并发执行；
kill -9 掉一个 worker，它 PEL 里没确认的任务会被别的 worker 用 XAUTOCLAIM 接走。

容错演示（kill -9 一个 worker，任务被存活 worker 接走）：
    docker compose exec worker python tools/demo_failover.py            # 自动演示
    docker compose exec worker python tools/demo_failover.py --manual   # 手动：打印 pid 后等你亲手 kill -9

为什么独立进程而不是塞进 web：
    web 只负责「提交 + 查状态」，worker 只负责「消费 + 执行」，
    两者解耦才能独立扩容 worker 数量，也才能演示「kill -9 一个 worker 任务不丢」。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import web  # noqa: F401 —— 复用 web 里初始化好的 graph / CACHE / JOBS（含 handler 注册）


def main() -> int:
    web.JOBS.run_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
