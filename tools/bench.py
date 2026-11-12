"""压测 / 故障注入 / 重试对照实验。

四组实验，全部真跑（真 HTTP、真 Redis、真 LLM、真数据）：
  A. 缓存命中路径并发压测（uvicorn + httpx，32 并发）
  B. 冷路径端到端耗时（真 LLM，3 次）
  C. 故障注入：Redis 全挂 / LLM 全挂，看服务是否仍可用
  D. 重试对照：260 次调用、40% 注入失败率，无重试 vs 指数退避重试

用法（项目根目录）：
    python -B tools/bench.py
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

import web  # noqa: E402
from omics_agent import cache as cache_mod  # noqa: E402
from omics_agent import metrics, resilience  # noqa: E402

PORT = int(os.environ.get("BENCH_PORT", "8011"))
BASE = f"http://127.0.0.1:{PORT}"
QUESTION = "帮我找两组样本的差异表达基因"
REPORT = "【压测用报告正文】\n显著差异基因 2801 个；肿瘤组上调 1402、下调 1399；通路富集见附表。"


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


def start_server():
    config = uvicorn.Config(web.app, host="127.0.0.1", port=PORT, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE}/healthz", timeout=1.0).status_code in (200, 503):
                return server
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("服务没起来")


def token_headers():
    token = web.create_token(1, "bench")
    return {"Authorization": f"Bearer {token}"}


def bench_cache_hit(concurrency=32, total=512):
    key = cache_mod.build_key(QUESTION, "data")
    web.CACHE.set(key, REPORT, ttl=3600)
    headers = token_headers()
    body = {"user_request": QUESTION, "dataset_path": "data", "user_id": "bench"}

    # 预热：把连接池、中间件、JWT 校验都跑热
    with httpx.Client(timeout=30) as client:
        for _ in range(20):
            client.post(f"{BASE}/analyze", json=body, headers=headers)

    latencies, errors = [], []
    lock = threading.Lock()
    per_worker = total // concurrency

    def worker():
        with httpx.Client(timeout=60) as client:
            local_lat, local_err = [], []
            for _ in range(per_worker):
                start = time.perf_counter()
                try:
                    r = client.post(f"{BASE}/analyze", json=body, headers=headers)
                    elapsed = (time.perf_counter() - start) * 1000
                    if r.status_code != 200 or not r.json().get("from_cache"):
                        local_err.append(f"HTTP {r.status_code}")
                    else:
                        local_lat.append(elapsed)
                except Exception as exc:  # noqa: BLE001
                    local_err.append(type(exc).__name__)
            with lock:
                latencies.extend(local_lat)
                errors.extend(local_err)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start

    done = len(latencies)
    return {
        "concurrency": concurrency,
        "requests": total,
        "completed": done,
        "errors": len(errors),
        "error_samples": errors[:5],
        "wall_s": round(wall, 2),
        "throughput_rps": round(done / wall, 1) if wall else 0,
        "p50_ms": pct(latencies, 0.50),
        "p90_ms": pct(latencies, 0.90),
        "p95_ms": pct(latencies, 0.95),
        "p99_ms": pct(latencies, 0.99),
        "mean_ms": round(statistics.fmean(latencies), 2) if latencies else None,
    }


def bench_cold_path(runs=3):
    """冷路径真耗时：每次先清掉缓存 key，否则测到的是上一轮留下的命中结果。"""
    headers = token_headers()
    key = cache_mod.build_key(QUESTION, "data")
    out = []
    with httpx.Client(timeout=180) as client:
        for i in range(runs):
            web.CACHE.delete(key)  # 关键：不清缓存会测成"命中路径"，耗时 0.01s
            body = {"user_request": QUESTION, "dataset_path": "data", "user_id": "bench"}
            t0 = time.perf_counter()
            r = client.post(f"{BASE}/analyze", json=body, headers=headers)
            t1 = time.perf_counter()
            if r.status_code != 200:
                out.append({"status": r.status_code, "error": r.text[:200]})
                continue
            data = r.json()
            item = {"analyze_s": round(t1 - t0, 2), "status": data.get("status"),
                    "from_cache": bool(data.get("from_cache"))}
            if data.get("status") == "pending_confirm":
                t2 = time.perf_counter()
                rc = client.post(f"{BASE}/confirm", json={"thread_id": data["thread_id"], "approved": True},
                                 headers=headers)
                item["confirm_s"] = round(time.perf_counter() - t2, 2)
                item["report_chars"] = len(rc.json().get("report") or "") if rc.status_code == 200 else 0
                item["total_s"] = round(item["analyze_s"] + item["confirm_s"], 2)
            out.append(item)
    totals = [i["total_s"] for i in out if i.get("total_s")]
    return {"runs": out,
            "median_total_s": round(statistics.median(totals), 2) if totals else None,
            "max_total_s": max(totals) if totals else None,
            "all_cold": all(not i.get("from_cache") for i in out)}


class BrokenRedis:
    def get(self, *a, **kw):
        raise ConnectionError("redis down")

    def set(self, *a, **kw):
        raise ConnectionError("redis down")

    def delete(self, *a, **kw):
        raise ConnectionError("redis down")

    def ping(self):
        raise ConnectionError("redis down")


def bench_redis_down():
    """Redis 整体不可用：服务不能 500，必须降级到进程内缓存并继续返回报告。"""
    key = cache_mod.build_key(QUESTION + "（redis 故障注入）", "data")
    original = web.CACHE
    web.CACHE = cache_mod.ReportCache(redis_client=BrokenRedis(), connect=False)
    try:
        web.CACHE.set(key, REPORT, ttl=600)
        body = {"user_request": QUESTION + "（redis 故障注入）", "dataset_path": "data", "user_id": "bench"}
        start = time.perf_counter()
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{BASE}/analyze", json=body, headers=token_headers())
        elapsed = (time.perf_counter() - start) * 1000
        payload = r.json() if r.status_code == 200 else {}
        return {
            "status_code": r.status_code,
            "from_cache": payload.get("from_cache"),
            "cache_source": payload.get("cache_source"),
            "latency_ms": round(elapsed, 2),
            "report_ok": bool(payload.get("report")),
        }
    finally:
        web.CACHE = original


def bench_llm_down():
    """LLM 整体不可用：意图解析走关键词兜底、解读降级为统计摘要，端到端仍要出报告。"""
    import omics_agent.config as cfg
    import omics_agent.nodes as nodes

    def boom(*a, **kw):
        raise ConnectionError("llm down")

    orig_cfg, orig_nodes = cfg.get_model, nodes.get_model
    cfg.get_model = boom
    nodes.get_model = boom
    try:
        headers = token_headers()
        body = {"user_request": "帮我找两组样本的差异表达基因（llm 故障注入）",
                "dataset_path": "data", "user_id": "bench"}
        t0 = time.perf_counter()
        with httpx.Client(timeout=180) as client:
            r = client.post(f"{BASE}/analyze", json=body, headers=headers)
            if r.status_code != 200:
                return {"status_code": r.status_code, "error": r.text[:300]}
            data = r.json()
            report = data.get("report")
            if data.get("status") == "pending_confirm":
                rc = client.post(f"{BASE}/confirm",
                                 json={"thread_id": data["thread_id"], "approved": True}, headers=headers)
                report = rc.json().get("report") if rc.status_code == 200 else None
        return {
            "status_code": 200,
            "total_s": round(time.perf_counter() - t0, 2),
            "report_chars": len(report or ""),
            "degraded_but_served": bool(report),
        }
    finally:
        cfg.get_model = orig_cfg
        nodes.get_model = orig_nodes


def bench_retry_ab(trials=260, failure_rate=0.4, seed=7):
    """对照实验：同样 40% 的调用失败率，没有重试 vs 指数退避重试成功率。"""
    def make_fn(rng, state):
        def fn():
            state["calls"] += 1
            if rng.random() < failure_rate:
                raise ConnectionError("注入失败")
            return True
        return fn

    rows = {}
    for label, attempts in (("no_retry", 1), ("with_retry", 3)):
        rng = random.Random(seed)  # 两种策略用同一串随机数，保证可比
        state = {"calls": 0}
        ok = 0
        fn = make_fn(rng, state)
        for _ in range(trials):
            try:
                resilience.call_with_retry(fn, attempts=attempts, base_delay=0.0,
                                           max_delay=0.0, jitter=0.0, sleep=lambda s: None,
                                           label=f"bench_{label}")
                ok += 1
            except Exception:  # noqa: BLE001
                pass
        rows[label] = {"success": ok, "trials": trials,
                       "success_rate": round(100.0 * ok / trials, 1),
                       "llm_calls": state["calls"]}
    return rows


def main():
    results = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    server = start_server()
    try:
        print("[A] 缓存命中路径并发压测 ...")
        results["cache_hit_load"] = bench_cache_hit()
        print(json.dumps(results["cache_hit_load"], ensure_ascii=False))

        print("[C1] Redis 故障注入 ...")
        results["redis_down"] = bench_redis_down()
        print(json.dumps(results["redis_down"], ensure_ascii=False))

        print("[D] 重试对照实验 ...")
        results["retry_ab"] = bench_retry_ab()
        print(json.dumps(results["retry_ab"], ensure_ascii=False))

        print("[C2] LLM 故障注入（真实数据，会跑一遍差异+富集）...")
        results["llm_down"] = bench_llm_down()
        print(json.dumps(results["llm_down"], ensure_ascii=False))

        print("[B] 冷路径端到端（真 LLM，3 次）...")
        results["cold_path"] = bench_cold_path()
        print(json.dumps(results["cold_path"], ensure_ascii=False))
    finally:
        server.should_exit = True
        time.sleep(0.5)

    out = os.path.join(ROOT, "bench", "last_run.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
