"""限流层：Redis 令牌桶 + 滑动窗口（零新依赖，只用项目已有的 redis 客户端）。

为什么需要（每条对应一个真实的被滥用面）：
  1. /login 会做 bcrypt 校验，而 bcrypt 是**故意慢**的（单次约 100ms CPU）。
     不限流时，一份极便宜的 HTTP 请求就能换走一份昂贵的 CPU —— 少量并发就能把进程打满，
     所有正常请求一起变慢。这是"廉价入口换昂贵计算"的放大器，必须掐在入口；
  2. /register 同样要算一次 bcrypt 并往 MySQL 写一行，可以被脚本刷；
  3. /analyze 一次要跑约 25s 的图、烧 token，是成本最高、最容易被当成免费 API 的入口；
  4. /upload 单个文件允许 200MB，没有配额就能把磁盘写满。

为什么两种算法（不是一个够用吗）：
  - 令牌桶（token bucket）：有容量上限、按速率匀速回补，**允许短时突发**。
    适合 /login —— 真人登录就是"偶尔连点两下"，严格滑动窗口会让正常用户在
    "第 6 次点重试"时突然被卡住；令牌桶有"攒下的额度"，体验更贴近直觉，
    同时长期速率仍然被钉死（回补速率 = capacity / 窗口）；
  - 滑动窗口（sliding window）：严格"N 次 / M 时间"的硬上限，**没有突发**。
    适合 /analyze、/upload 这种按次数硬记账的入口 —— 10 次/分钟就该是 10 次，
    不该因为上一分钟没用过就允许这一分钟一口气跑 30 次。
  两者都只用 Redis 原生命令（HASH / ZSET），不引入任何新依赖。

为什么必须放 Redis 而不是进程内字典：
  web.py 按 uvicorn 多 worker 部署（WEB_CONCURRENCY / --workers），进程内计数每个 worker 各算一份，
  3 个 worker 就是 3 倍额度，限流形同虚设。Redis 在多 worker 之间共享，才谈得上"全局 N 次"。

为什么用 Lua 脚本：
  "读剩余额度 → 判断 → 扣减"必须原子。若用 Python 先 GET 再 SET，两个并发请求会读到同一个
  剩余数、都判断"还有"、都扣一次 → 实际放行数超过上限。整段逻辑交给 Redis（单线程）执行，
  这段窗口才不存在；顺带把一次限流的往返从 2~3 次 RTT 压成 1 次。

Redis 挂掉怎么办：**降级放行**并计 metric（ratelimit_degraded_total）。
  限流防的是"滥用"，可用性管的是"服务还能不能用"。Redis 挂的时候若因为限流把所有人挡在门外，
  就是把一个依赖故障放大成全站故障。降级放行 + 计数告警，是这里的正确取舍
  —— 与 omics_agent/cache.py 对 Redis 的态度保持一致，整条链路只有一种降级哲学。
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, NamedTuple

from omics_agent import metrics

LOGGER = logging.getLogger("omics.ratelimit")

# Redis key 前缀：便于在 redis-cli 里 KEYS rl:* 一眼看清限流相关键，也方便运维单独清理
KEY_PREFIX = "rl"

# 限流判定的重试时间下限（秒）：算出来是 0 也要回 1，避免客户端看到 Retry-After: 0 就立刻重试
MIN_RETRY_AFTER_S = 1

# 身份串长度上限：用户名/UA 之类的输入可能极长，直接拼进 redis key 会让 key 无限膨胀
MAX_IDENTITY_LEN = 96


class Decision(NamedTuple):
    """一次限流判定的结果。调用方（web.py）只需要 allowed / retry_after / degraded。"""

    allowed: bool
    limit: int           # 本次判定用的额度，便于日志和响应排查
    retry_after: int     # 建议客户端等待的秒数（仅 allowed=False 时有意义）
    remaining: int       # 剩余额度（降级时为 limit）
    degraded: bool       # True = 这次判定没敢依赖 Redis（放行），需要监控


# ── Lua：令牌桶 ───────────────────────────────────────────────────────────
# KEYS[1] = 桶 key
# ARGV[1] = capacity（容量，允许的最大突发）  ARGV[2] = per_seconds（多少秒回补满）
# ARGV[3] = now_ms（客户端时间，毫秒）        ARGV[4] = cost（本次要消耗几个令牌）
# 返回 {allowed(0/1), 剩余令牌(整数), 建议等待秒数}
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local per_seconds = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local rate = capacity / per_seconds
if rate <= 0 then rate = 0.000001 end

local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts = tonumber(bucket[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now_ms
end
if now_ms < ts then ts = now_ms end
tokens = math.min(capacity, tokens + ((now_ms - ts) / 1000.0) * rate)

local allowed = 0
local retry = 0
if tokens >= cost then
  allowed = 1
  tokens = tokens - cost
else
  retry = math.ceil((cost - tokens) / rate)
end

redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', tostring(now_ms))
redis.call('PEXPIRE', key, math.ceil((capacity / rate) * 2000))
return {allowed, math.floor(tokens), retry}
"""

# ── Lua：滑动窗口 ─────────────────────────────────────────────────────────
# KEYS[1] = 窗口 zset key
# ARGV[1] = limit（窗口内最多几次）  ARGV[2] = window_ms（窗口长度，毫秒）
# ARGV[3] = now_ms                   ARGV[4] = cost     ARGV[5] = member 前缀（保证唯一）
# 返回 {allowed(0/1), 剩余次数, 建议等待秒数}
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local member = ARGV[5]

redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms - window_ms)
local used = redis.call('ZCARD', key)
local allowed = 0
local remaining = 0
local retry = 0
if used + cost <= limit then
  for i = 1, cost do
    redis.call('ZADD', key, now_ms, member .. ':' .. i)
  end
  redis.call('PEXPIRE', key, window_ms)
  allowed = 1
  remaining = limit - used - cost
else
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  if oldest[2] then
    retry = math.ceil(((tonumber(oldest[2]) + window_ms) - now_ms) / 1000.0)
  end
  if retry < 1 then retry = 1 end
end
return {allowed, remaining, retry}
"""


# ── Redis 客户端（可注入，便于测试与"复用缓存那条连接"）──────────────────
_client: Any = None
_client_resolved = False
_client_lock = threading.Lock()


def configure(client: Any) -> None:
    """由 web.py 注入已有的 redis 客户端（通常是 cache 那条连接）。

    为什么允许注入而不是自己再建一条：web.py 的 CACHE 已经建好了一条带超时和
    decode_responses 的连接；限流再开一条就是多一份连接池和一份配置漂移的机会。
    传入 None 也是合法输入 —— 代表"本进程没有 Redis"，限流自动降级放行。
    """
    global _client, _client_resolved
    with _client_lock:
        _client = client
        _client_resolved = True


def _build_client() -> Any:
    """没被 configure 过时的兜底：按环境变量自建一条短超时连接；失败返回 None。"""
    try:
        import redis  # 局部导入：与 cache.py 一致，测试/离线环境不装也能 import 本模块

        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            # 限流不该给请求加长尾：Redis 慢的时候超时短一点，快速降级比拖住请求更好
            socket_timeout=float(os.getenv("RATELIMIT_REDIS_TIMEOUT_S", "0.5")),
            socket_connect_timeout=float(os.getenv("RATELIMIT_REDIS_TIMEOUT_S", "0.5")),
            decode_responses=True,
        )
        client.ping()
        return client
    except Exception as exc:  # noqa: BLE001 - Redis 不可用是预期内的降级路径
        metrics.incr("ratelimit_redis_unavailable_total")
        LOGGER.warning("限流无法连接 Redis，将降级放行：%s: %s", type(exc).__name__, exc)
        return None


def get_client() -> Any:
    """惰性拿客户端：只在第一次真正需要限流时才建连接（import 期不连外部依赖）。"""
    global _client, _client_resolved
    if _client_resolved:
        return _client
    with _client_lock:
        if not _client_resolved:
            _client = _build_client()
            _client_resolved = True
    return _client


def reset_client() -> None:
    """清掉已解析的客户端（测试用：模拟"从没有 Redis"切到"有 Redis"）。"""
    global _client, _client_resolved
    with _client_lock:
        _client = None
        _client_resolved = False


# ── 基础设施 ──────────────────────────────────────────────────────────────
def _enabled() -> bool:
    """总开关。留这个开关是为了"线上出事要立刻止血"时能不改代码先关掉限流。"""
    return os.getenv("RATELIMIT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def _identity(raw: Any) -> str:
    """身份串归一化：去空白、统一小写、截断。

    为什么统一小写：不归一的话 "Alice" / "alice" / "ALICE " 是三个桶，
    同一个账号只要换大小写就能绕过按用户名的限流。
    为什么截断：用户名是客户端给的，超长输入直接拼进 redis key 会让 key 膨胀。
    """
    text = str(raw or "").strip().lower().replace("|", "/")
    return text[:MAX_IDENTITY_LEN] or "unknown"


def _degraded(limit: int) -> Decision:
    """Redis 用不了时的统一出口：放行 + 计数。

    注意这里**不抛异常**：限流组件坏了不能把业务请求一起拖死，
    但降级必须可见（metric + 日志），否则"限流悄悄失效"比"限流太严"更危险。
    """
    metrics.incr("ratelimit_degraded_total")
    return Decision(allowed=True, limit=limit, retry_after=0, remaining=limit, degraded=True)


def _run_script(script: str, key: str, args: list) -> list | None:
    """执行 Lua，返回三元组结果；任何 Redis 侧问题都返回 None（交给调用方降级）。"""
    client = get_client()
    if client is None:
        return None
    try:
        return client.eval(script, 1, key, *args)
    except Exception as exc:  # noqa: BLE001 - 超时/连不上/加载失败都走同一条降级路径
        metrics.incr("ratelimit_redis_error_total")
        LOGGER.warning("限流脚本执行失败，本次降级放行：%s: %s", type(exc).__name__, exc)
        return None


# ── 两种算法的对外形态 ────────────────────────────────────────────────────
def token_bucket(namespace: str, identity: str, capacity: int, per_seconds: float, cost: int = 1) -> Decision:
    """令牌桶：capacity 个突发的额度，per_seconds 秒匀速回补满。"""
    capacity = max(1, int(capacity))
    if not _enabled():
        return Decision(True, capacity, 0, capacity, False)

    key = f"{KEY_PREFIX}:tb:{namespace}:{_identity(identity)}"
    now_ms = int(time.time() * 1000)
    raw = _run_script(
        _TOKEN_BUCKET_LUA, key, [capacity, float(per_seconds), now_ms, max(1, int(cost))]
    )
    if not raw:
        return _degraded(capacity)

    allowed = bool(int(raw[0]))
    remaining = max(0, int(raw[1]))
    retry_after = max(0, int(raw[2]))
    if not allowed:
        metrics.incr(f"ratelimit_limited_{namespace}_total")
        retry_after = max(MIN_RETRY_AFTER_S, retry_after)
    return Decision(allowed, capacity, retry_after if not allowed else 0, remaining, False)


def sliding_window(namespace: str, identity: str, limit: int, window_seconds: float, cost: int = 1) -> Decision:
    """滑动窗口：window_seconds 秒内最多 limit 次，没有突发额度。"""
    limit = max(1, int(limit))
    if not _enabled():
        return Decision(True, limit, 0, limit, False)

    key = f"{KEY_PREFIX}:sw:{namespace}:{_identity(identity)}"
    now_ms = int(time.time() * 1000)
    # member 必须唯一：同一毫秒内并发请求若用相同 member，ZADD 会互相覆盖，次数就漏记了
    member = f"{now_ms}-{uuid.uuid4().hex[:12]}"
    raw = _run_script(
        _SLIDING_WINDOW_LUA,
        key,
        [limit, int(max(1.0, float(window_seconds)) * 1000), now_ms, max(1, int(cost)), member],
    )
    if not raw:
        return _degraded(limit)

    allowed = bool(int(raw[0]))
    remaining = max(0, int(raw[1]))
    retry_after = max(0, int(raw[2]))
    if not allowed:
        metrics.incr(f"ratelimit_limited_{namespace}_total")
        retry_after = max(MIN_RETRY_AFTER_S, retry_after)
    return Decision(allowed, limit, retry_after if not allowed else 0, remaining, False)


# ── 各路由的具体策略 ──────────────────────────────────────────────────────
def check_login(ip: str, username: str) -> Decision:
    """登录：按 (IP, 用户名) 一个桶，再叠一层按 IP 的更宽桶。

    为什么两个维度都要（只做一个是漏洞）：
      - 只按用户名：攻击者拿字典轮着试不同用户名，每个用户名都不超限，等于没有防护；
      - 只按 IP：分布式/Botnet 从很多 IP 打同一个用户名时完全不设防。
    两个都设，才同时挡住"单点爆破多账号"和"多 IP 爆破单账号"。
    额度按分钟折算成回补速率（capacity/60 每秒），所以它是"允许连点几下，但长期被钉在 N 次/分钟"。
    """
    per_minute = int(os.getenv("RATELIMIT_LOGIN_PER_MIN", "10"))
    ip_per_minute = int(os.getenv("RATELIMIT_LOGIN_IP_PER_MIN", "30"))

    # 先判细粒度（按账号），再判粗粒度（按 IP）：细粒度的 429 更精准，也不会先把共享 IP 的同事牵连进去
    user_decision = token_bucket("login", f"{_identity(ip)}|{_identity(username)}", per_minute, 60)
    if not user_decision.allowed:
        return user_decision
    return token_bucket("login_ip", _identity(ip), ip_per_minute, 60)


def check_register(ip: str) -> Decision:
    """注册：按 IP 严格限速。

    注册 = 一次 bcrypt + 一行 MySQL 写入，天然是批量刷号的目标。
    用硬上限（而不是令牌桶）是因为注册没有"突发"的正当理由：正常人在一小时内不会注册 10 个账号。

    默认值取 10 次/小时，而不是更漂亮的 3 次/小时：这个数还要能容纳项目自带的接口测试
    （每跑一轮 tests/ 就会注册 1~2 个随机用户，跑三四轮就到 3 了）。
    默认值把正常开发流程卡死，只会让人把限流关掉 —— 那是比限得松更糟的结果。
    """
    return sliding_window("register", _identity(ip), int(os.getenv("RATELIMIT_REGISTER_PER_HOUR", "10")), 3600)


def check_analyze(user_id: Any) -> Decision:
    """/analyze 与 /analyze_async：按 user_id 严格限速（成本最高的入口）。

    必须按 user_id 而不是 IP：分析成本是"账号"造出来的（会烧 token、占线程 25s），
    同一个办公室/公司出口 IP 下几十个人共用一个 IP，按 IP 限流会误伤正常用户。
    异步入口共用同一个桶：否则"同步被限就改走异步"就是一条现成的绕过路径。
    """
    return sliding_window("analyze", _identity(user_id), int(os.getenv("RATELIMIT_ANALYZE_PER_MIN", "10")), 60)


def check_upload(user_id: Any) -> Decision:
    """上传：按 user_id 限速（默认 20 次/小时）。

    上传的代价是磁盘 IO + 空间，累积效应比单次更值得关注，所以用小时级硬上限。
    """
    return sliding_window("upload", _identity(user_id), int(os.getenv("RATELIMIT_UPLOAD_PER_HOUR", "20")), 3600)
