"""FastAPI 后端：鉴权 / 上传 / 分析（同步 + 异步）/ 缓存 / 健康检查 / 指标。

本轮加固（每条都对应一个真实故障）：
  1. 缓存 key 带上数据集指纹 —— 修掉"换一份数据、问同一句话，却拿到上一份报告"的正确性 bug；
  2. Redis 主存 + 进程内兜底 + TTL 抖动 + single-flight —— Redis 挂掉服务仍可用；
     同一句话并发进来只跑一次图（一次图实测 ~25s，重复计算就是白烧线程与 token）；
  3. 长任务异步化（/analyze_async + /jobs/{id}）—— 25s 的图执行不再占着 HTTP 连接；
  4. /healthz + /metrics + X-Request-ID —— 出问题能回答"哪个依赖挂了、哪条请求、慢在哪一步"；
  5. 上传与 dataset_path 加固 —— 扩展名白名单、大小上限、路径必须落在 data/ 内
     （原来 dataset_path 完全由客户端给，可以指到任意目录）。

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
import logging
import shutil
import time
import uuid
from typing import List

import bcrypt
import jwt
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
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
    save_history,
    update_history_report,
    update_password,
)
from omics_agent import jobs as jobs_mod
from omics_agent import metrics
from omics_agent.cache import ReportCache, build_key
from omics_agent.graph import build_graph

LOGGER = logging.getLogger("omics.web")


def _async_jobs_available() -> bool:
    """异步任务是否可用。

    异步任务会把图跑到 **worker 进程**里，跑到 interrupt 挂起时，检查点只存在
    那台 worker 的 checkpointer 上；/confirm 打到 backend 进程，必须有共享的
    持久化检查点才取得到这个会话。InMemorySaver 下这条路是断的，必须往前拒，
    而不是让用户拿到一个永远确认不了的 job。
    """
    return os.getenv("USE_POSTGRES", "").lower() == "true"

app = FastAPI(title="多组学数据分析智能体", version="2.0")

# 造图成本高（编译 + 连 checkpointer），服务常驻，造一次反复用
graph = build_graph()

# 缓存与任务注册表：进程内单例
CACHE = ReportCache()
redis_client = CACHE.redis  # 兼容旧代码里对 redis_client 的引用
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

# ============ 第 2 块：JWT 配置常量 ============
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # 没配密钥时不要静默用 None（jwt.encode 会 TypeError）；
    # 退化成进程级随机密钥并明确告警：重启后旧 token 失效，但不至于服务起不来。
    SECRET_KEY = uuid.uuid4().hex
    print("[warn] 未配置 SECRET_KEY，已生成进程级临时密钥；重启后 token 将失效")
ALGORITHM = "HS256"
# 单位是分钟：默认 1440 = 24 小时。（原值 60*60 是 3600 分钟 = 60 小时，名实不符）
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# 数据集根目录：dataset_path 必须落在它下面（防路径穿越）
ALLOWED_DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "data"))

# 上传约束
UPLOAD_MAX_BYTES = int(os.getenv("UPLOAD_MAX_MB", "200")) * 1024 * 1024
UPLOAD_MAX_FILES = int(os.getenv("UPLOAD_MAX_FILES", "20"))
ALLOWED_SUFFIXES = (".csv", ".tsv", ".txt")

# single-flight 里"等别人算完"的最长等待（秒）：超时宁可自己再算一遍，也不把请求挂死
CACHE_WAIT_S = float(os.getenv("CACHE_WAIT_S", "180"))
# 人工确认后最多续跑几轮：防止 resume 立刻又 interrupt 造成死循环
MAX_RESUME_ROUNDS = int(os.getenv("MAX_RESUME_ROUNDS", "5"))


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
@app.middleware("http")
async def observability(request, call_next):
    """给每条请求打 X-Request-ID，并记录按"路由模板"聚合的耗时与状态码。"""
    rid = request.headers.get("x-request-id") or metrics.new_request_id()
    request.state.request_id = rid  # 异常处理器要用：contextvar 在中间件退出时已被重置
    token = metrics.set_request_id(rid)
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
def register(req: RegisterRequest):
    if get_user_by_username(req.username):  # 先查有没有同名
        raise HTTPException(status_code=400, detail="用户名已存在")
    hashed = hash_password(req.password)
    create_user(username=req.username, hashed_password=hashed)
    return {"message": "注册成功"}


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/login")
def login(req: LoginRequest):
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
    if not files:
        raise HTTPException(status_code=400, detail="没有收到文件")
    if len(files) > UPLOAD_MAX_FILES:
        raise HTTPException(status_code=400, detail=f"一次最多上传 {UPLOAD_MAX_FILES} 个文件")

    # 必须用 ALLOWED_DATA_ROOT 拼绝对路径：相对 CWD 落盘时，
    # 换一个工作目录启动就会"上传成功但分析一律 400（不在 data/ 内）"
    task_dir = os.path.join(ALLOWED_DATA_ROOT, "upload", str(uuid.uuid4()))
    os.makedirs(task_dir, exist_ok=True)

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
                    if written > UPLOAD_MAX_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{name} 超过上限 {UPLOAD_MAX_BYTES // (1024 * 1024)}MB",
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
    """把客户端给的路径解析成绝对路径，并强制落在 data/ 之内。"""
    path = os.path.abspath(raw or "data")
    root = os.path.normcase(ALLOWED_DATA_ROOT)
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
    with metrics.timer("graph.phase1_ms"):
        result = graph.invoke(initial, config=config)

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
    try:
        with metrics.timer("graph.phase2_ms"):
            result = graph.invoke(Command(resume={"approved": req.approved}), config=config)
            rounds = 0
            while result.get("__interrupt__") and rounds < MAX_RESUME_ROUNDS:
                # 还中断就继续续跑；但必须有轮数上限，否则异常输入能卡死一个线程
                rounds += 1
                metrics.incr("graph.resume_round_total")
                result = graph.invoke(Command(resume={"approved": req.approved}), config=config)
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
