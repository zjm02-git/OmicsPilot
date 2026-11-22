"""FastAPI 后端：鉴权 / 上传 / 分析（同步 + 异步）/ 缓存 / 健康检查 / 指标。

本轮加固（每条都对应一个真实故障）：
  1. 缓存 key 带上数据集指纹 —— 修掉"换一份数据、问同一句话，却拿到上一份报告"的正确性 bug；
  2. Redis 主存 + 进程内兜底 + TTL 抖动 + single-flight —— Redis 挂掉服务仍可用；
     同一句话并发进来只跑一次图（一次图实测 ~25s，重复计算就是白烧线程与 token）；
  3. 长任务异步化（/analyze_async + /jobs/{id}）—— 25s 的图执行不再占着 HTTP 连接；
  4. /healthz + /metrics + X-Request-ID —— 出问题能回答"哪个依赖挂了、哪条请求、慢在哪一步"；
  5. 上传与 dataset_path 加固 —— 扩展名白名单、大小上限、路径必须落在 data/ 内
     （原来 dataset_path 完全由客户端给，可以指到任意目录）。

本轮再补的三块（同样"每条对应一个真实故障"）：
  6. 启动期副作用收进 lifespan —— 建表与 checkpointer.setup() 不再在 import 期把进程带走；
     进程活着，/healthz 的 503 才可能被编排系统看到（否则"MySQL 挂了就摘流量"只是句口号）；
  7. 限流 —— /login /register 按 IP（挡住"廉价请求换 bcrypt 昂贵计算"）、/analyze /upload 按 user_id；
     超限 429 + Retry-After；Redis 不可用时降级放行并计 metric；
  8. 上传请求级总量配额 + 过期目录清理；request_id 注入每一行日志（原来只存在 metrics 里，日志里查不到）。

接口契约保持不变（frontend/app.py 依赖）：
  /register /login /me /user /change_password
  /upload → {dataset_path}
  /analyze → {status: pending_confirm|done, thread_id, qc_report, group_a, group_b, report}
  /confirm → {report}
  /history → 列表
"""

# ============ 第 0 块：先把 .env 读进来 ============
# 必须在任何 os.getenv 之前：原版本把 load_dotenv() 放在 redis 客户端之后，
# REDIS_HOST / REDIS_PORT / SECRET_KEY 实际都没读到，静默走了默认值。
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ============ 第 1 块：导入库 + 创建应用 ============
import asyncio
import logging
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import List

import bcrypt
import jwt
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.types import Command
from pydantic import BaseModel

from db import (
    create_user,
    delete_user,
    get_history_by_user,
    get_user_by_username,
    history_owner_id,
    init_db,
    save_history,
    update_history_report,
    update_password,
)
from omics_agent import jobs as jobs_mod
from omics_agent import metrics
from omics_agent import ratelimit
from omics_agent.cache import ReportCache, build_key
from omics_agent.graph import build_graph

LOGGER = logging.getLogger("omics.web")

# ============ 第 0.5 块：让 request_id 落进每一行日志（T6）============
# 现状问题：middleware 把 request_id 塞进了 metrics 的 contextvar，响应头里也有 X-Request-ID，
# 但**没有任何一行日志打印过它**。于是线上出现"某个请求 500"时，
# 用户报来一串 request_id，我们却无法在日志里把它定位到那一条 —— 只能靠时间戳瞎猜。
# 这里补两件事：
#   1. RequestIdFilter：每条 LogRecord 从 contextvar 取出 request_id 写进 record.request_id；
#   2. RequestIdFormatter：格式串里固定带 %(request_id)s，保证它真被打印出来。
# 为什么两个都要：只加 Filter 不换 Formatter 时，字段进了 record 但格式串里没有它，
# 看起来"加了却没生效"；只换 Formatter 不加 Filter 时，record 上没有这个属性会抛 KeyError。
LOG_FORMAT = "%(asctime)s %(levelname)-5s [%(request_id)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RequestIdFilter(logging.Filter):
    """把当前上下文的 request_id 注入每条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        # 已有 request_id 就不覆盖：允许调用方显式传入（比如后台任务自报一条业务 id）
        if not getattr(record, "request_id", None):
            try:
                record.request_id = metrics.get_request_id()
            except Exception:  # noqa: BLE001 - 日志链路永远不能因为取不到 id 而抛异常
                record.request_id = "-"
        return True


class RequestIdFormatter(logging.Formatter):
    """兜底格式化器：即使某个 handler 上漏挂了 Filter，也不会因为缺字段而丢整行日志。"""

    def format(self, record: logging.LogRecord) -> str:
        if not getattr(record, "request_id", None):
            try:
                record.request_id = metrics.get_request_id()
            except Exception:  # noqa: BLE001
                record.request_id = "-"
        return super().format(record)


# 需要覆盖的 logger：root 管我们自己打的日志；uvicorn.* 是它启动后自己建 handler 的那几个，
# 不一起处理的话"请求日志里没有 request_id"依旧存在，排查时还是要靠猜。
_LOG_TARGETS = ("", "uvicorn", "uvicorn.error", "uvicorn.access", "omics")
_request_id_filter = RequestIdFilter()


def setup_request_id_logging() -> None:
    """给 root / uvicorn / omics 的 handler 挂上 Filter 并换 Formatter（幂等，可反复调用）。

    为什么要求幂等：import 期调一次（覆盖我们自己打的日志），lifespan 启动段再调一次
    —— 因为 uvicorn 是在 import 之后才安装它自己的 handler 的，只在 import 期处理会漏掉它们。
    """
    if not logging.getLogger().handlers:
        # 没人配置过日志时给一个默认输出，否则 INFO 级别的日志会静默消失
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    for name in _LOG_TARGETS:
        target = logging.getLogger(name)
        for handler in list(target.handlers):
            if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
                handler.addFilter(_request_id_filter)
            if not isinstance(handler.formatter, RequestIdFormatter):
                handler.setFormatter(RequestIdFormatter(LOG_FORMAT, LOG_DATE_FORMAT))
        # logger 自身也挂一份：即使某条日志被 propagation 到上层处理，记录上也已经有 request_id
        if not any(isinstance(f, RequestIdFilter) for f in target.filters):
            target.addFilter(_request_id_filter)


# import 期就先装一次：worker 进程（worker.py 只是 import web，不会跑 lifespan）也能拿到 request_id
setup_request_id_logging()


def _async_jobs_available() -> bool:
    """异步任务是否可用。

    异步任务会把图跑到 **worker 进程**里，跑到 interrupt 挂起时，检查点只存在
    那台 worker 的 checkpointer 上；/confirm 打到 backend 进程，必须有共享的
    持久化检查点才取得到这个会话。InMemorySaver 下这条路是断的，必须往前拒，
    而不是让用户拿到一个永远确认不了的 job。
    """
    return os.getenv("USE_POSTGRES", "").lower() == "true"

# ============ 第 1.4 块：惰性造图（T1 的后一半）============
# 造图成本高（编译 + 连 checkpointer），服务常驻，所以造一次反复用。
# 但它同时是**第二处 import 期重副作用**：config.get_checkpointer() 在 USE_POSTGRES=true 时
# 会连 Postgres 并跑 checkpointer.setup()（建表 + 迁移）。PG 没起来时，原来那句
# graph = build_graph() 在模块顶层直接抛异常，进程照样起不来 —— 和 db.py 建表是同一个病。
# 改法：惰性 + 可重试。import 期只留一个占位，第一次真正要用（/analyze、/confirm、lifespan
# 启动段）时才造；造不出来就回 503，而不是把进程带走。
_GRAPH = None
_GRAPH_LOCK = threading.Lock()


def get_graph():
    """拿到图；必要时现造一个。造失败把异常抛给调用方（由 _require_graph 转成 503）。"""
    global _GRAPH, graph
    if _GRAPH is not None:
        return _GRAPH
    with _GRAPH_LOCK:  # 双检加锁：并发请求不能同时造出两张图（会各自连一套 checkpointer）
        if _GRAPH is None:
            _GRAPH = build_graph()
            graph = _GRAPH  # 兼容旧的模块属性引用（web.graph）
    return _GRAPH


def graph_ready() -> bool:
    """/healthz 用：图有没有造好。造好才意味着 checkpointer 真的连上了。"""
    return _GRAPH is not None


def _require_graph():
    """给路由用：图不可用时回 503（依赖没就绪），而不是 500（代码有 bug）。

    503 与 500 的区别对编排系统很关键：503 = "过会儿再来"，500 = "这段代码坏了，快来人"。
    把"依赖没起来"报成 500，等于每天制造假报警，把真正的 500 埋掉。
    """
    try:
        return get_graph()
    except Exception as exc:  # noqa: BLE001 - 造图失败的根因（PG 连不上）必须落日志
        metrics.incr("graph_unavailable_total")
        LOGGER.warning("图不可用（检查点/模型依赖未就绪）：%s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=503,
            detail="分析引擎尚未就绪（检查点数据库或模型依赖不可用），请稍后重试",
        ) from exc


# ============ 第 1.5 块：创建应用 + lifespan（T1 启动期副作用）============
# 原版本把两件重量级副作用放在 **import 期**：
#   1) db.py 的 Base.metadata.create_all(engine) —— 连 MySQL 建表；
#   2) build_graph() 里的 checkpointer.setup()     —— 连 PG 建表。
# 后果：MySQL / PG 只要有一个没起来，进程**根本起不来**，于是 /healthz 永远回不了 503 ——
# 而 README 承诺的正是"MySQL 挂了就 503，让编排系统摘流量"。进程直接死掉时，
# 编排系统看到的是"启动失败 / 反复重启"，不是"健康但依赖不可用"，两者处置完全不同。
# 改法：两个副作用挪进 lifespan，失败只告警并按退避重试，服务照常启动、如实报 503。
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动：跑完（允许失败的）就绪动作 → 起周期清理任务；关闭：收回清理任务。"""
    await _startup_init()
    # 上传清理必须随服务生死：放独立守护线程的话，uvicorn --reload 反复重启会累积出多个清理线程，
    # 而且进程退出后它们还在跑。挂在 lifespan 上，finally 里 cancel 就干净了。
    cleanup_task = asyncio.create_task(_upload_cleanup_loop(), name="upload-cleanup")
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(title="多组学数据分析智能体", version="2.0", lifespan=lifespan)

# 兼容旧的模块属性引用；真正的取用一律走 get_graph() / _require_graph()，此处可能是 None
graph = None

# 缓存与任务注册表：进程内单例
CACHE = ReportCache()
redis_client = CACHE.redis  # 兼容旧代码里对 redis_client 的引用

# 限流复用缓存那条 Redis 连接：同一份配置、同一个连接池，不额外开一条（少一处配置漂移）。
# Redis 不可用时 redis_client 是 None，ratelimit 会降级放行并记 metric —— 见 omics_agent/ratelimit.py。
ratelimit.configure(redis_client)

# 熔断器的失败计数也放到同一条 Redis 上：否则 3 个 worker 就是 3 份独立计数，
# "总闸"在多进程下形同虚设（进程级熔断只能防止单个 worker 把网关打死）。
# 传 None 或 Redis 不可用时，CircuitBreaker 自动退回进程内计数并记 metric。
from omics_agent import resilience as resilience_mod  # noqa: E402 - 必须在 CACHE 之后接线

resilience_mod.configure_breaker_backend(redis_client)
# 队列要长阻塞读（XREADGROUP BLOCK），不能用缓存那条 socket_timeout=2s 的连接（2s 一到就被打断），
# 单独建一条不限 socket 超时的连接给 worker 消费用。
_queue_redis = None
if redis_client is not None:
    try:
        import redis as _redis

        _queue_redis = _redis.Redis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            socket_timeout=None,          # 阻塞读：等多久由 BLOCK 决定，不受 socket 超时打断
            socket_connect_timeout=5.0,
            decode_responses=True,
        )
        _queue_redis.ping()
    except Exception:  # noqa: BLE001 - Redis 真不可用时队列退化为不可用，submit 会明确报错
        _queue_redis = None
JOBS = jobs_mod.JobRegistry(
    redis_client=_queue_redis,
    ttl=int(os.getenv("JOB_TTL_S", "3600")),
    # 必须大于单步 LLM 超时（resilience 默认 90s）：否则任务还在跑就被判超时
    visible_timeout_s=float(os.getenv("JOB_VISIBLE_TIMEOUT_S", "180")),
    max_retries=int(os.getenv("JOB_MAX_RETRIES", "3")),
)

# ============ 第 2 块：JWT 配置常量（T4：多 worker 下没密钥必须拒绝启动）============
ALGORITHM = "HS256"

# 为什么这里要分单 worker / 多 worker 两种处理：
#   没配 SECRET_KEY 时，原来的做法是"生成一个进程级随机密钥 + 打一行告警"。
#   单 worker 下这只是一点小麻烦（重启后旧 token 失效，用户重新登录即可）。
#   多 worker 下（uvicorn --workers N / gunicorn / 容器拉起多个副本）**每个 worker 各自随机**，
#   于是 A worker 签发的 token 到 B worker 验签必然失败 —— 表现为**随机 401**：
#   同一个用户刷新几次，有时能用有时不能用，取决于请求落到哪个 worker，极难定位，
#   而且日志里只会显示"验签失败"，看起来像客户端在伪造 token。
#   这种"配置缺失导致的事故"，宁可启动时就明确失败（把问题钉在部署那一刻），
#   也不要带着病跑起来再随机 401。
_WORKER_ENV_VARS = ("WEB_CONCURRENCY", "UVICORN_WORKERS")
MIN_SECRET_KEY_LEN = 32  # HS256 建议至少 32 字节随机串（openssl rand -hex 32）


def detect_workers() -> tuple[int, str]:
    """返回 (worker 数, 来源环境变量名)。两个变量都没设 / 写错时按单进程处理。"""
    for name in _WORKER_ENV_VARS:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            continue
        try:
            count = int(raw)
        except ValueError:
            # 写错了就当没配：宁可少拦一次（告警 + 随机密钥），
            # 也不要因为一个拼错的环境变量让整个服务起不来
            LOGGER.warning("%s=%r 不是合法整数，忽略", name, raw)
            continue
        if count > 1:
            return count, name
    return 1, ""


def resolve_secret_key(secret: str | None, workers: int, source: str = "") -> str:
    """决定本进程用的 JWT 密钥。多 worker 且没配就直接抛错（= 启动失败）。

    单独抽成函数是为了可测：不用真的起多个 worker，也能验证"多 worker 无密钥 → 抛错"这条规则。
    """
    secret = (secret or "").strip()
    if secret:
        if len(secret) < MIN_SECRET_KEY_LEN:
            LOGGER.warning(
                "SECRET_KEY 只有 %d 字节，短于建议的 %d 字节；"
                "HS256 密钥太短可被离线爆破（建议 openssl rand -hex 32）",
                len(secret), MIN_SECRET_KEY_LEN,
            )
        return secret

    if workers > 1:
        # 直接抛异常，uvicorn 会以非 0 退出码结束 —— 这就是"启动失败"，错误信息里写清了怎么修
        raise RuntimeError(
            f"检测到多 worker 部署（{source}={workers}）但没有配置 SECRET_KEY："
            "每个 worker 会各自生成随机密钥，A worker 签发的 token 到 B worker 必然验签失败"
            "（随机 401，极难定位）。请设置环境变量 SECRET_KEY=<至少 32 字节的随机串> 后重启；"
            "本地单进程调试请用 uvicorn --workers 1（或直接 uvicorn web:app）。"
        )

    LOGGER.warning(
        "未配置 SECRET_KEY，已生成进程级临时密钥：重启后旧 token 全部失效；"
        "多 worker 部署下会直接拒绝启动（避免随机 401）"
    )
    return uuid.uuid4().hex


_WORKERS, _WORKER_SOURCE = detect_workers()
SECRET_KEY = resolve_secret_key(os.getenv("SECRET_KEY"), _WORKERS, _WORKER_SOURCE)
# 单位是分钟：默认 1440 = 24 小时。（原值 60*60 是 3600 分钟 = 60 小时，名实不符）
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# 数据集根目录：dataset_path 必须落在它下面（防路径穿越）
# T5：用 realpath 而不是 abspath。abspath 只做"拼接 + 规范化"，**不会解析 junction / symlink**：
# data/ 下被放一个指向 C:\Windows 的 junction 时，abspath 看到的仍是 "...\data\junc"（校验通过），
# 但真实读写发生在 C:\Windows —— 守的是一个假路径。realpath 会把链接一路解析到底，
# 比较的才是"文件到底在哪儿"。
ALLOWED_DATA_ROOT = os.path.realpath(os.path.join(BASE_DIR, "data"))

# 上传约束
UPLOAD_MAX_BYTES = int(os.getenv("UPLOAD_MAX_MB", "200")) * 1024 * 1024
UPLOAD_MAX_FILES = int(os.getenv("UPLOAD_MAX_FILES", "20"))
ALLOWED_SUFFIXES = (".csv", ".tsv", ".txt")

# ── T3：上传的"请求级总量"配额 + 过期清理 ──
# UPLOAD_MAX_MB 限的是**单个文件**。20 个文件 × 200MB ≈ 4GB，一次请求就能把盘写满；
# 而且这 4GB 本身也是一次成本极高的"合法"操作。所以再加一道按**请求合计**的上限（默认 400MB）。
UPLOAD_TOTAL_MAX_BYTES = int(os.getenv("UPLOAD_TOTAL_MAX_MB", "400")) * 1024 * 1024
# data/upload/ 以前**从不清理**：只有上传失败回滚才删，成功上传的目录永远留着。
# 跑几天就能把盘写满，而磁盘满会先表现为"MySQL 写失败 / 日志写不出来 / 服务莫名 500"，
# 极难追回到上传目录这条线上。这里按 mtime 保留 N 小时，由 lifespan 的周期任务扫。
UPLOAD_RETENTION_HOURS = float(os.getenv("UPLOAD_RETENTION_HOURS", "24"))
UPLOAD_CLEANUP_INTERVAL_S = float(os.getenv("UPLOAD_CLEANUP_INTERVAL_S", "3600"))
UPLOAD_ROOT = os.path.join(ALLOWED_DATA_ROOT, "upload")

# 启动期副作用的重试策略：MySQL / PG 没起来时按 1s、2s、3s 线性退避重试，
# 用完之后**照常启动**（不再把进程带走），由 /healthz 如实报告 503。
STARTUP_RETRY_ATTEMPTS = int(os.getenv("STARTUP_RETRY_ATTEMPTS", "3"))
STARTUP_RETRY_DELAY_S = float(os.getenv("STARTUP_RETRY_DELAY_S", "1"))

# single-flight 里"等别人算完"的最长等待（秒）：超时宁可自己再算一遍，也不把请求挂死
CACHE_WAIT_S = float(os.getenv("CACHE_WAIT_S", "180"))
# 人工确认后最多续跑几轮：防止 resume 立刻又 interrupt 造成死循环
MAX_RESUME_ROUNDS = int(os.getenv("MAX_RESUME_ROUNDS", "5"))


# ============ 第 2.5 块：服务层基础设施（限流落地 / 启动重试 / 上传清理）============
def _client_ip(request: Request) -> str:
    """取客户端 IP，作为"按 IP 限流"的维度。

    为什么不读 X-Forwarded-For：那个头客户端可以随手伪造，不校验就采信等于把限流关掉
    —— 打一枪换一个"IP"，每个 IP 都不超限。真在反向代理后面部署时，正确做法是
    明确配置"可信代理层数"，再从 XFF 右边开始取第 N 跳；那需要一份代理拓扑配置，不该靠猜。
    当前实现宁可把所有人都算成代理 IP（限流变严），也不接受一个可伪造的维度。
    """
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "unknown"


def _enforce_rate_limit(decision: ratelimit.Decision, route: str) -> None:
    """把限流判定翻译成 HTTP 语义：放行走，超限回 429 + Retry-After。

    为什么一定要带 Retry-After：只回 429 而不说"等多久"，带自动重试的客户端（很多 SDK 默认就是）
    往往立刻重试，把一次限流放大成一轮重试风暴 —— 限流反而成了压垮服务的最后一脚。
    """
    if decision.allowed:
        return
    LOGGER.warning(
        "限流命中 route=%s limit=%s retry_after=%ss degraded=%s",
        route, decision.limit, decision.retry_after, decision.degraded,
    )
    raise HTTPException(
        status_code=429,
        detail=f"请求过于频繁，请 {decision.retry_after} 秒后重试",
        headers={"Retry-After": str(decision.retry_after)},
    )


def cleanup_stale_uploads(retention_hours: float | None = None) -> int:
    """删除 data/upload/ 下 mtime 超过保留期的任务目录，返回删掉的个数。

    为什么必须清：见 UPLOAD_RETENTION_HOURS 上方的说明 —— 不清理就是一条慢性的磁盘泄漏。
    为什么删之前还要用 realpath 复核一遍：这是个**删除**操作，而 data/ 是上传内容能影响的地方。
    如果那里被放了 junction，路径看着仍在上传根目录内，真实目标却可能在别处；
    删除类操作只能删"解析后依然在上传根目录之内"的目录（与 T5 是同一条道理）。
    """
    root = os.path.realpath(UPLOAD_ROOT)
    if not os.path.isdir(root):
        return 0
    hours = UPLOAD_RETENTION_HOURS if retention_hours is None else float(retention_hours)
    deadline = time.time() - max(hours, 0.0) * 3600.0
    root_key = os.path.normcase(root)

    # 先收集再删除：Windows 上边遍历边 rmtree 容易撞到目录句柄占用（scandir 迭代器握着句柄）
    targets: list[str] = []
    with os.scandir(root) as it:
        for entry in it:
            try:
                if not entry.is_dir():
                    continue
                real = os.path.realpath(entry.path)
                if os.path.normcase(real) == root_key:
                    continue
                try:
                    inside = os.path.commonpath([root_key, os.path.normcase(real)]) == root_key
                except ValueError:  # 跨盘符（junction 指到别的盘）：commonpath 直接抛，必然在外面
                    inside = False
                if not inside:
                    metrics.incr("upload_cleanup_skipped_total")
                    LOGGER.warning("跳过疑似跳出上传根目录的条目：%s -> %s", entry.path, real)
                    continue
                if entry.stat().st_mtime > deadline:
                    continue
                targets.append(entry.path)
            except OSError as exc:  # 单个条目读不到不该让整轮清理失败
                metrics.incr("upload_cleanup_error_total")
                LOGGER.warning("检查上传目录条目失败 %s：%s", getattr(entry, "path", "?"), exc)

    removed = 0
    for path in targets:
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as exc:
            metrics.incr("upload_cleanup_error_total")
            LOGGER.warning("删除过期上传目录失败 %s：%s", path, exc)
    if removed:
        metrics.incr("upload_cleanup_removed_total", float(removed))
    return removed


async def _upload_cleanup_loop() -> None:
    """lifespan 里的周期任务：每 UPLOAD_CLEANUP_INTERVAL_S 扫一次过期上传目录。

    为什么用 to_thread：os.scandir / shutil.rmtree 是阻塞系统调用，直接写在协程里会卡住事件循环
    —— 清理一个大目录的那几百毫秒里，所有请求都会被冻住。
    """
    while True:
        try:
            await asyncio.sleep(UPLOAD_CLEANUP_INTERVAL_S)
            removed = await asyncio.to_thread(cleanup_stale_uploads)
            if removed:
                LOGGER.info("上传目录清理完成：删除 %d 个过期子目录", removed)
        except asyncio.CancelledError:
            raise  # 取消是真信号，必须往上冒，否则 lifespan 收不干净它
        except Exception as exc:  # noqa: BLE001 - 清理失败绝不能把周期任务打死
            metrics.incr("upload_cleanup_error_total")
            LOGGER.warning("上传目录清理失败：%s: %s", type(exc).__name__, exc)


async def _startup_step(name: str, fn) -> bool:
    """跑一个启动期就绪动作，允许失败并按线性退避重试；返回最终是否成功。

    注意"失败也不抛"：就绪动作失败不该让进程死，只该让 /healthz 变成 503。
    这正是把"存活（liveness）"和"就绪（readiness）"分开的意义。
    """
    attempts = max(1, STARTUP_RETRY_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            await asyncio.to_thread(fn)  # 建表要连 MySQL，是阻塞调用，必须挪出事件循环
            if attempt > 1:
                LOGGER.info("[startup] %s 第 %d 次重试成功", name, attempt)
            return True
        except Exception as exc:  # noqa: BLE001 - 依赖没起来是预期场景，重试即可
            metrics.incr(f"startup_{name}_failed_total")
            LOGGER.warning(
                "[startup] %s 失败（第 %d/%d 次）：%s: %s",
                name, attempt, attempts, type(exc).__name__, exc,
            )
            if attempt < attempts:
                await asyncio.sleep(STARTUP_RETRY_DELAY_S * attempt)
    LOGGER.error("[startup] %s 最终失败：服务照常启动，/healthz 将如实返回 503", name)
    return False


async def _startup_init() -> None:
    """lifespan 启动段：把"必须连上外部依赖"的两件事做完，但**失败不阻塞启动**。

    顺序有讲究：先建表（MySQL），再造图（checkpointer 要连 PG）。两个都失败也照常启动 ——
    进程活着、指标和日志都在，运维才看得到"是哪一个依赖挂了"。
    顺便跑一次上传清理：重启后第一件事就回收磁盘，不必等第一个清理周期。
    """
    setup_request_id_logging()  # uvicorn 的 handler 是在 import 之后装的，这里补挂一次
    await _startup_step("db_init", init_db)
    await _startup_step("graph_build", get_graph)
    removed = await asyncio.to_thread(cleanup_stale_uploads)
    if removed:
        LOGGER.info("启动期清理：删除 %d 个过期上传目录", removed)


# ============ 第 3 块：密码哈希（bcrypt）============
def hash_password(password: str) -> str:
    # 盐 = 随机字符：同一密码每次哈希都不同，防彩虹表撞库
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# ============ 第 4 块：签发 token ============
def create_token(user_id: int, username: str) -> str:
    payload = {
        "sub": str(user_id),  # sub = subject，JWT 标准字段
        "username": username,
        "exp": int(time.time() + ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ============ 第 5 块：可观测（请求 ID + 按路由耗时 + 状态码）============
# 这里同时是"request_id 能落进日志"的唯一入口（T6）：
# 日志 Filter 不自己生成 id，它只从 metrics 的 contextvar 里取 —— 而 contextvar 就是在这里被设置的。
# 所以顺序很重要：必须在 call_next 之前 set，之后的每条日志（包括异常栈）才带得上这个 id。
@app.middleware("http")
async def observability(request, call_next):
    """给每条请求打 X-Request-ID，并记录按"路由模板"聚合的耗时与状态码。"""
    # 上游（网关/前端）已经带了 id 就沿用：一次用户点击跨多个服务时，整条链路才能串起来
    rid = request.headers.get("x-request-id") or metrics.new_request_id()
    request.state.request_id = rid  # 异常处理器要用：contextvar 在中间件退出时已被重置
    token = metrics.set_request_id(rid)  # ← 从这里开始，本请求产生的每条日志都会带 [rid]
    start = time.perf_counter()
    status_code = 500
    response = None
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:  # noqa: BLE001 - 先记账再抛，否则 500 的耗时统计会丢
        metrics.incr("http_unhandled_error_total")
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # 用路由模板（/jobs/{job_id}）而不是原始路径，否则每个 id 都会变成一条独立指标
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"
        metrics.observe_ms(f"http.{request.method}.{path}", elapsed_ms)
        metrics.incr(f"http_status_{status_code}_total")
        # T6 的"清理"这一半：contextvar 会跟着复用的事件循环/线程活到下一个请求上去，
        # 不 reset 就会出现"这条日志的 request_id 属于上一条请求"—— 串号的 id 比没有 id 更害人，
        # 它会把人引到完全错误的请求上。所以放在 finally 里，出错也照样清。
        metrics.reset_request_id(token)

    response.headers["X-Request-ID"] = rid
    response.headers["X-Elapsed-Ms"] = f"{elapsed_ms:.0f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """兜底：漏网的异常也要回一段可定位的 JSON，而不是光秃秃的 Internal Server Error。

    响应里带 request_id，用户报错时报这一串就能在日志里定位到同一条请求。
    """
    metrics.incr("unhandled_exception_total")
    rid = getattr(request.state, "request_id", None) or metrics.get_request_id()
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务内部错误，请稍后重试",
            "error_type": type(exc).__name__,
            "request_id": rid,
        },
        headers={"X-Request-ID": rid or ""},
    )


# ============ 第 6 块：注册 / 登录 / 验签 ============
security = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    username: str
    password: str


@app.post("/register")
def register(req: RegisterRequest, request: Request):
    # 限流必须在最前面：bcrypt 计算和写库都在后面，等它们跑完再拒，CPU 和时间已经花掉了
    _enforce_rate_limit(ratelimit.check_register(_client_ip(request)), "register")
    if get_user_by_username(req.username):  # 先查有没有同名
        raise HTTPException(status_code=400, detail="用户名已存在")
    hashed = hash_password(req.password)
    create_user(username=req.username, hashed_password=hashed)
    return {"message": "注册成功"}


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/login")
def login(req: LoginRequest, request: Request):
    # 必须在校验之前限流：bcrypt.checkpw 是**故意慢**的（单次约 100ms CPU），
    # 不限流时攻击者用极便宜的请求换极贵的计算 —— 等于把我们变成自己的 CPU DoS 帮凶。
    # 维度是 (IP, 用户名) 再加一层 IP：前者挡"单点爆破多账号"，后者挡"多 IP 爆破单账号"。
    _enforce_rate_limit(ratelimit.check_login(_client_ip(request), req.username), "login")
    user = get_user_by_username(req.username)
    # 用户不存在 / 密码不对统一回同一句，避免被枚举出"哪些用户名存在"
    if user is None or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"access_token": create_token(user.id, user.username), "token_type": "Bearer"}


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="没带登记牌")
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:  # 是 InvalidTokenError 的子类，要先捕获
        raise HTTPException(status_code=401, detail="登机牌过期了")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="登机牌是假的或被动过")
    return {"user_id": int(payload["sub"]), "username": payload["username"]}


@app.get("/me")
def read_me(user: dict = Depends(get_current_user)):
    return user


# ============ 第 7 块：账号管理 ============
@app.delete("/user")
def delete_account(user: dict = Depends(get_current_user)):
    delete_user(user["username"])
    return {"message": "账号已删除"}


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


@app.post("/change_password")
def change_password(req: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    do_user = get_user_by_username(user["username"])
    if do_user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if not verify_password(req.old_password, do_user.hashed_password):
        raise HTTPException(status_code=401, detail="旧密码不对")
    update_password(user["username"], hash_password(req.new_password))
    return {"message": "密码修改成功"}


# ============ 第 8 块：上传（白名单 + 大小上限 + 文件名净化）============
def _safe_filename(raw: str | None) -> str:
    """只留文件名本身：去掉目录成分，挡掉 ../../ 这类路径穿越。"""
    name = os.path.basename((raw or "").replace("\\", "/")).strip()
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="文件名非法")
    if not name.lower().endswith(ALLOWED_SUFFIXES):
        raise HTTPException(
            status_code=400,
            detail=f"只接受 {'/'.join(ALLOWED_SUFFIXES)} 文件，收到：{name}",
        )
    return name


@app.post("/upload")
def upload_files(files: List[UploadFile] = File(), user: dict = Depends(get_current_user)):
    # 按 user_id 限流：上传占的是磁盘和带宽，是"累积起来"才出事的成本，所以用小时级硬上限
    _enforce_rate_limit(ratelimit.check_upload(user["user_id"]), "upload")
    if not files:
        raise HTTPException(status_code=400, detail="没有收到文件")
    if len(files) > UPLOAD_MAX_FILES:
        raise HTTPException(status_code=400, detail=f"一次最多上传 {UPLOAD_MAX_FILES} 个文件")

    # 请求级总量预检：UploadFile.size 由 multipart 解析时给出，已知大小就先拒，
    # 别等把 400MB 真的落盘之后再说"超限"——落盘的 IO 和空间已经付出去了。
    declared_total = sum(int(getattr(f, "size", None) or 0) for f in files)
    if declared_total > UPLOAD_TOTAL_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"本次上传合计 {declared_total // (1024 * 1024)}MB，"
                f"超过单次请求上限 {UPLOAD_TOTAL_MAX_BYTES // (1024 * 1024)}MB"
            ),
        )

    # 必须用绝对根目录落盘（UPLOAD_ROOT 由 ALLOWED_DATA_ROOT 派生）：相对 CWD 落盘时，
    # 换一个工作目录启动就会"上传成功但分析一律 400（不在 data/ 内）"。
    # 与清理任务共用同一个根，才不会出现"清理扫 A 目录、上传写 B 目录"的空转。
    task_dir = os.path.join(UPLOAD_ROOT, str(uuid.uuid4()))
    os.makedirs(task_dir, exist_ok=True)

    total_written = 0  # 整个请求的累计字节数（跨文件），对应 UPLOAD_TOTAL_MAX_MB
    try:
        for f in files:
            name = _safe_filename(f.filename)
            declared = getattr(f, "size", None)  # 已知大小就先拒，别把超大文件落盘再报错
            if declared is not None and declared > UPLOAD_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{name} 超过上限 {UPLOAD_MAX_BYTES // (1024 * 1024)}MB",
                )
            written = 0
            target = os.path.join(task_dir, name)
            with open(target, "wb") as out:
                while True:  # 分块写：不把整个文件读进内存，也能中途拦住超大文件
                    chunk = f.file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    total_written += len(chunk)
                    if written > UPLOAD_MAX_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{name} 超过上限 {UPLOAD_MAX_BYTES // (1024 * 1024)}MB",
                        )
                    # 单文件没超、但请求合计超了也要拦：20 个 190MB 的文件每个都"合法"，
                    # 加起来却是 3.8GB —— 这一层防的正是"每个都合规、整体离谱"。
                    if total_written > UPLOAD_TOTAL_MAX_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"本次上传合计超过 {UPLOAD_TOTAL_MAX_BYTES // (1024 * 1024)}MB 上限"
                                f"（UPLOAD_TOTAL_MAX_MB），在 {name} 处中止"
                            ),
                        )
                    out.write(chunk)
            if written == 0:
                raise HTTPException(status_code=400, detail=f"{name} 是空文件")
    except Exception:
        shutil.rmtree(task_dir, ignore_errors=True)  # 失败不留半成品目录占盘
        raise

    metrics.incr("upload_total")
    return {"dataset_path": task_dir}


# ============ 第 9 块：分析核心（缓存 + single-flight + 图执行）============
class AnalizeRequest(BaseModel):
    user_request: str
    dataset_path: str = "data"
    user_id: str = "request-01"
    use_cache: bool = True  # 基准测试要测"纯冷路径"时置 false


class ConfirmRequest(BaseModel):
    thread_id: str
    approved: bool


def _validate_dataset_path(raw: str) -> str:
    """把客户端给的路径解析成**真实**绝对路径，并强制落在 data/ 之内。

    T5：这里从 abspath 换成 realpath。两者都返回绝对路径，区别在"是否解析链接"：
      - abspath("data/link")  -> <项目根>/data/link   字符串上看确实在 data/ 里，校验通过；
      - realpath("data/link") -> C:/Windows          当 link 是指向外部的 junction/符号链接时，
                                                     文件真实读写发生在 data/ 之外。
    也就是说，原来的校验守的是"字符串路径"，而文件系统按"真实位置"办事 ——
    data/ 下只要存在一个 junction（上传目录、挂载盘、开发时图省事建的链接都可能带来），
    整道防线就被绕开了。
    改用 realpath 后比较的是解析到底的真实路径；对没有链接的普通目录，
    realpath 与 abspath 结果一致，因此原有语义（必须在 data/ 内、必须存在）完全保留。
    """
    path = os.path.realpath(raw or "data")
    root = os.path.normcase(ALLOWED_DATA_ROOT)  # 根也必须是 realpath（见常量定义处）
    target = os.path.normcase(path)
    try:
        common = os.path.commonpath([root, target])
    except ValueError:  # 连盘符都不同（如 D: 与 C:），commonpath 直接抛
        common = ""
    if target != root and common != root:
        raise HTTPException(status_code=400, detail="数据集路径必须位于 data/ 目录内")
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"数据集目录不存在：{raw}")
    return path


def _run_graph(asg: AnalizeRequest, dataset_path: str, history_user_id: int, cache_key: str) -> dict:
    """真正跑一次图，返回可直接回给前端的响应体。"""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "user_request": asg.user_request,
        "dataset_path": dataset_path,
        # 归属只认 JWT 解出来的 user_id：请求体里的 user_id 任何登录用户都能伪造，
        # 用它分桶等于允许别人读写你的长期记忆。
        "user_id": str(history_user_id),
    }

    jobs_mod.progress("质检 + 人工确认")
    # 惰性取图：造不出来（PG 不可用）就回 503，既不是 500，也不会把进程带走
    engine = _require_graph()
    with metrics.timer("graph.phase1_ms"):
        result = engine.invoke(initial, config=config)

    if result.get("__interrupt__"):
        info = result["__interrupt__"][0].value
        _remember_pending(thread_id, cache_key)  # confirm 之后用它回填缓存
        _safe_save_history(
            user_id=history_user_id,
            thread_id=thread_id,
            user_request=asg.user_request,
            status="pending",
        )
        return {
            "status": "pending_confirm",
            "thread_id": thread_id,
            "qc_report": info.get("qc_report"),
            "group_a": info.get("group_a"),
            "group_b": info.get("group_b"),
        }

    final_report = result.get("final_report") or ""
    _safe_save_history(
        user_id=history_user_id,
        thread_id=thread_id,
        user_request=asg.user_request,
        status="done",
        final_report=final_report,
    )
    body = {"status": "done", "report": final_report, "thread_id": thread_id}
    if result.get("degraded"):
        # 降级产出（模型不可用时只剩统计摘要）必须让调用方知道，前端才好提示"建议稍后重试"
        body["degraded"] = True
        body["degraded_stages"] = list(result.get("degraded_stages") or [])
    return body


# 与 db.py 的列宽对齐：超长会被 MySQL 严格模式拒绝，整条历史静默消失（连 pending 行都没有）
HISTORY_MAX_REQUEST = 500
HISTORY_MAX_REPORT = 10000


def _safe_save_history(**kwargs) -> None:
    """历史写库失败不能吞掉用户的报告：算都已经算完了，报告必须能拿走。"""
    req = kwargs.get("user_request")
    if isinstance(req, str) and len(req) > HISTORY_MAX_REQUEST:
        LOGGER.warning("user_request 超长已截断 len=%d", len(req))
        metrics.incr("history_truncated_request_total")
        kwargs["user_request"] = req[:HISTORY_MAX_REQUEST]
    rep = kwargs.get("final_report")
    if isinstance(rep, str) and len(rep) > HISTORY_MAX_REPORT:
        LOGGER.warning("final_report 超长已截断 len=%d", len(rep))
        metrics.incr("history_truncated_report_total")
        kwargs["final_report"] = rep[:HISTORY_MAX_REPORT]
    try:
        save_history(**kwargs)
    except Exception:  # noqa: BLE001
        metrics.incr("history_write_error_total")


def _run_and_cache(asg: AnalizeRequest, dataset_path: str, key: str, history_user_id: int) -> dict:
    body = _run_graph(asg, dataset_path, history_user_id, key)
    if body.get("status") == "done" and body.get("report"):
        # 只缓存完整报告：pending 还没算完；降级报告（模型挂了只有统计摘要）写进去
        # 会让后续 24 小时的所有同问请求都拿到这份残缺内容。
        if body.get("degraded"):
            metrics.incr("cache_skip_degraded_total")
        else:
            CACHE.set(key, body["report"])
            metrics.incr("cache_write_total")
    return body


def _cached_body(report: str, source: str, key: str) -> dict:
    return {
        "status": "done",
        "report": report,
        "from_cache": True,
        "cache_source": source,
        "cache_key": key,
    }


# HITL 会把"算到一半"的请求挂起，报告在 /confirm 才生成。
# 这里把缓存 key 暂存 2 小时，confirm 完成后回填，重复提问才能命中。
PENDING_TTL = 2 * 3600


def _remember_pending(thread_id: str, cache_key: str) -> None:
    CACHE.set(f"pending:{thread_id}", cache_key, ttl=PENDING_TTL)


def _pop_pending(thread_id: str) -> str | None:
    key = CACHE.get(f"pending:{thread_id}")
    if key:
        CACHE.delete(f"pending:{thread_id}")
    return key


def _core_analyze(asg: AnalizeRequest, history_user_id: int) -> dict:
    """缓存命中 → 直接返回；未命中 → 抢计算权，抢不到就等别人的结果。"""
    dataset_path = _validate_dataset_path(asg.dataset_path)
    key = build_key(asg.user_request, dataset_path, user_id=history_user_id)

    if not asg.use_cache:
        metrics.incr("cache_bypass_total")
        return _run_graph(asg, dataset_path, history_user_id, key)

    value, source = CACHE.get_source(key)
    if value:
        metrics.incr("cache_hit_total")
        return _cached_body(value, source, key)
    metrics.incr("cache_miss_total")

    with CACHE.inflight(key) as leader:
        if leader:
            # double-check：抢到锁之前，可能刚好有人写完了
            again, source2 = CACHE.get_source(key)
            if again:
                metrics.incr("cache_hit_total")
                metrics.incr("cache_singleflight_dedup_total")
                return _cached_body(again, source2, key)
            return _run_and_cache(asg, dataset_path, key, history_user_id)

        # 别人正在算：等他算完复用同一份结果（并发击穿合并）
        metrics.incr("cache_singleflight_wait_total")
        waited = CACHE.wait_for(key, timeout=CACHE_WAIT_S)
        if waited:
            metrics.incr("cache_hit_total")
            return _cached_body(waited, "wait", key)
        # 等超时（对端失败或太慢）：自己算一遍，宁可重复也不挂死服务
        metrics.incr("cache_singleflight_fallback_total")
        return _run_and_cache(asg, dataset_path, key, history_user_id)


def _handle_analyze_task(payload: dict) -> dict:
    """worker 侧执行：反序列化参数，跑一次分析（在独立 worker 进程里执行）。"""
    asg = AnalizeRequest(**payload["asg"])
    metrics.set_request_id(metrics.new_request_id())
    return _core_analyze(asg, payload["user_id"])


# 注册任务类型 → 处理函数（worker import web 时执行到这里，自动完成注册）
JOBS.register("analyze", _handle_analyze_task)


@app.post("/analyze")
def analyze(asg: AnalizeRequest, user: dict = Depends(get_current_user)):
    """同步分析：跑完才返回。适合前端愿意等，或已命中缓存的场景。"""
    # 按 user_id 限流（不是按 IP）：分析成本是"账号"造出来的（烧 token、占线程约 25s），
    # 而同一个出口 IP 后面往往是几十个正常用户，按 IP 限会误伤。
    _enforce_rate_limit(ratelimit.check_analyze(user["user_id"]), "analyze")
    with metrics.timer("api.analyze_ms"):
        body = _core_analyze(asg, user["user_id"])
    body["request_id"] = metrics.get_request_id()
    return body


@app.post("/confirm")
def confirm(req: ConfirmRequest, user: dict = Depends(get_current_user)):
    """人工确认后续跑（HITL 第二段）。"""
    # 资源级鉴权：thread_id 在历史表里属于谁，就只能谁确认。
    # 没有这条记录 = 不是这个用户的会话（或会话已过期），同样按不存在处理。
    owner_id = history_owner_id(req.thread_id)
    if owner_id is None or str(owner_id) != str(user["user_id"]):
        metrics.incr("confirm_access_denied_total")
        raise HTTPException(status_code=404, detail="会话不存在或已失效")

    config = {"configurable": {"thread_id": req.thread_id}}
    # 取图放在 try 之外：_require_graph 抛的 503 不能被下面的 except 吞成 400
    #（否则"依赖没就绪"会被说成"会话已失效"，用户和运维都被误导）
    engine = _require_graph()
    try:
        with metrics.timer("graph.phase2_ms"):
            result = engine.invoke(Command(resume={"approved": req.approved}), config=config)
            rounds = 0
            while result.get("__interrupt__") and rounds < MAX_RESUME_ROUNDS:
                # 还中断就继续续跑；但必须有轮数上限，否则异常输入能卡死一个线程
                rounds += 1
                metrics.incr("graph.resume_round_total")
                result = engine.invoke(Command(resume={"approved": req.approved}), config=config)
    except Exception as exc:  # noqa: BLE001
        metrics.incr("confirm_error_total")
        raise HTTPException(status_code=400, detail=f"确认失败，会话可能已失效：{exc}") from exc

    if result.get("__interrupt__"):
        metrics.incr("confirm_max_rounds_total")
        raise HTTPException(status_code=409, detail="该分析仍在等待人工确认，请重试或取消")

    final_report = result.get("final_report") or ""
    update_history_report(
        thread_id=req.thread_id, final_report=final_report, user_id=user["user_id"]
    )

    # 回填缓存：这条路径的报告在 confirm 才生成，不回填的话"同一问题第二次提交"
    # 永远不会命中缓存（差异分析主路径就是这样漏掉的）。
    # 但降级报告不能回填 —— 否则一次模型抖动会污染 24 小时内的所有同问请求。
    degraded = bool(result.get("degraded"))
    pending_key = _pop_pending(req.thread_id)
    if pending_key and final_report:
        if degraded:
            metrics.incr("cache_skip_degraded_total")
        else:
            CACHE.set(pending_key, final_report)
            metrics.incr("cache_write_total")
            metrics.incr("cache_backfill_from_confirm_total")

    body = {"report": final_report, "request_id": metrics.get_request_id()}
    if degraded:
        body["degraded"] = True
        body["degraded_stages"] = list(result.get("degraded_stages") or [])
    return body


# ============ 第 10 块：异步任务（提交 → 轮询）============
@app.post("/analyze_async")
def analyze_async(asg: AnalizeRequest, user: dict = Depends(get_current_user)):
    """提交长任务：写状态 + 入 Redis Stream 队列，立刻返回 job_id，前端轮询 /jobs/{id}。"""
    # 与 /analyze 共用同一个桶（namespace 都是 analyze）：异步入口算的是同一份钱和同一份 CPU，
    # 不共用的话"同步被限就改走异步"就是一条现成的绕过路径。
    _enforce_rate_limit(ratelimit.check_analyze(user["user_id"]), "analyze_async")
    if not _async_jobs_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "异步分析需要持久化检查点：请设置 USE_POSTGRES=true 并启动 Postgres。"
                "（图会在 worker 进程中断等待人工确认，进程内检查点在 /confirm 时取不到；"
                "不开启时请改用同步的 /analyze）"
            ),
        )
    try:
        job_id = JOBS.submit(
            "analyze",
            {"asg": asg.model_dump(), "user_id": user["user_id"]},
            owner=user["user_id"],
        )
    except Exception as exc:  # noqa: BLE001 - Redis 不可用时队列不可用
        raise HTTPException(status_code=503, detail=f"任务队列不可用：{exc}") from exc
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, user: dict = Depends(get_current_user)):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    # 资源级鉴权：job 里没有归属信息或归属不是当前用户，一律当"不存在"处理
    # （用 404 而不是 403，避免把"这个 job_id 真的存在"这个信息泄露出去）
    if str(job.get("user_id")) != str(user["user_id"]):
        metrics.incr("job_access_denied_total")
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return job


# ============ 第 11 块：历史 ============
@app.get("/history")
def get_history(user: dict = Depends(get_current_user)):
    histories = get_history_by_user(user["user_id"])
    return [
        {
            "id": h.id,
            "user_request": h.user_request,
            "final_report": h.final_report,
            "status": h.status,
            "created_at": h.created_at,
        }
        for h in histories
    ]


# ============ 第 12 块：健康检查 + 指标 ============
def _check_mysql() -> bool:
    try:
        from sqlalchemy import text

        from db import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


@app.get("/healthz")
def healthz():
    """依赖健康检查：区分"服务活着"和"依赖可用"。

    Redis 不可用算 degraded 而不是 down —— 缓存层会自动降级到进程内，业务仍可用；
    MySQL 不可用则鉴权/历史全废，直接 503，让编排系统摘流量。
    """
    # 探活接口自己绝不能因为依赖挂了而 500：那会让编排系统分不清
    # "服务挂了" 和 "依赖挂了"，而后者恰恰是最需要它给出答案的时候。
    def _probe(fn, default=False):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("healthz probe failed: %s", exc)
            return default

    def _safe(fn, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("healthz section failed: %s", exc)
            return default

    checks = {
        "redis": _probe(CACHE.ping, False),
        "mysql": _probe(_check_mysql, False),
        "graph_checkpointer": "postgres" if os.getenv("USE_POSTGRES", "").lower() == "true" else "memory",
        # 图有没有造好 = checkpointer 到底通不通（T1 之后建表可能失败，这一项才看得出来）
        "graph_ready": graph_ready(),
        "llm_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        "async_jobs_available": _async_jobs_available(),
    }
    critical_ok = bool(checks["mysql"])
    status = "ok" if critical_ok and checks["redis"] else ("degraded" if critical_ok else "down")
    body = {
        "status": status,
        "checks": checks,
        "cache": _safe(CACHE.stats, {}),
        "jobs": _safe(JOBS.stats, {"available": False}),
        "uptime_s": _safe(lambda: metrics.snapshot()["uptime_s"], None),
    }
    return JSONResponse(status_code=200 if critical_ok else 503, content=body)


@app.get("/metrics")
def get_metrics(user: dict = Depends(get_current_user)):
    """进程内指标：计数器 + 各路由 P50/P90/P95/P99。"""
    from omics_agent.dataio import csv_cache_info

    snap = metrics.snapshot()
    snap["cache"] = CACHE.stats()
    snap["jobs"] = JOBS.stats()
    snap["csv_cache"] = csv_cache_info()
    return snap
