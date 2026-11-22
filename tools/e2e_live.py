"""真模型端到端烟测：真 DeepSeek + 真 TCGA 数据，跑 /analyze → /confirm 全流程。

为什么单独一个脚本、不塞进 unittest？
  - tests/ 里所有用例都用 FakeModel：可离线、可复现、零费用，适合每次改动都跑；
  - 但"假模型能跑通"≠"真模型能跑通"：真实网关会超时/限流/返回格式不同。
    这个脚本把真实链路跑一遍，打印分段耗时与报告长度，用于演示与上线前自检。

默认不在测试里跑，是因为它**会花真钱**（一次 1 次意图解析 + 1 次解读 ≈ 几分钱），
而且受网络影响不稳定 —— 这两点都不该拖累日常回归。

用法（在项目根目录）：
    python tools/e2e_live.py
    python tools/e2e_live.py --question "帮我找两组样本的差异表达基因" --dataset data

输出：一段 JSON（analyze_s / confirm_s / total_s / report_chars / degraded），
      以及报告开头 400 字，方便肉眼确认"是不是真模型写的、四章节齐不齐"。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # web.py 的 load_dotenv 依赖项目根目录的 .env


def main() -> int:
    ap = argparse.ArgumentParser(description="真模型端到端烟测（会产生少量 token 费用）")
    ap.add_argument("--question", default="帮我找两组样本的差异表达基因")
    ap.add_argument("--dataset", default="data", help="相对项目根的数据集目录")
    ap.add_argument("--user", default="", help="复用已有账号；留空则临时建一个")
    args = ap.parse_args()

    from fastapi.testclient import TestClient
    from sqlalchemy import text

    import web
    from db import engine
    from omics_agent.cache import build_key

    client = TestClient(web.app)

    # ── 1. 账号（临时账号，跑完删掉，不污染 users 表）──────────────
    username = args.user or f"e2e_{uuid.uuid4().hex[:8]}"
    password = "E2eLive!2026"
    client.post("/register", json={"username": username, "password": password})  # 已存在也算正常
    r = client.post("/login", json={"username": username, "password": password})
    if r.status_code != 200:
        print("登录失败：", r.status_code, r.text)
        return 1
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    user_id = client.get("/me", headers=headers).json()["user_id"]

    dataset = os.path.abspath(os.path.join(ROOT, args.dataset))
    # 缓存 key 含用户维度（报告依赖该用户自己的阈值偏好），必须用登录后的 user_id 算
    cache_key = build_key(args.question, dataset, user_id=user_id)
    web.CACHE.delete(cache_key)  # 必须清掉，否则可能命中旧缓存、根本没调模型

    summary: dict = {
        "question": args.question,
        "dataset": dataset,
        "model": os.getenv("MODEL_NAME", "(config 默认)"),
        "llm_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
    }
    thread_id = None
    try:
        # ── 2. /analyze：意图解析 + 质检 + 挂起等人工确认 ──────────
        t0 = time.perf_counter()
        r = client.post(
            "/analyze",
            json={
                "user_request": args.question,
                "dataset_path": dataset,
                "user_id": username,
                "use_cache": False,  # 强制走真路径
            },
            headers=headers,
        )
        t_analyze = time.perf_counter() - t0
        summary["analyze_status"] = r.status_code
        summary["analyze_s"] = round(t_analyze, 2)
        if r.status_code != 200:
            print("analyze 失败：", r.status_code, r.text[:500])
            return 1

        body = r.json()
        summary["status"] = body.get("status")
        summary["qc_report"] = body.get("qc_report")
        summary["groups"] = [body.get("group_a"), body.get("group_b")]
        if body.get("status") != "pending_confirm":
            # 意图被判成 other / 质检没过，都会直接出报告 —— 也算跑通，只是没走到解读
            print(json.dumps({**summary, "report_head": (body.get("report") or "")[:400]},
                             ensure_ascii=False, indent=2))
            return 0

        thread_id = body["thread_id"]

        # ── 3. /confirm：差异分析 + 富集 + LLM 解读（真模型耗时主要在这里）──
        t1 = time.perf_counter()
        r2 = client.post(
            "/confirm", json={"thread_id": thread_id, "approved": True}, headers=headers
        )
        t_confirm = time.perf_counter() - t1
        summary["confirm_status"] = r2.status_code
        summary["confirm_s"] = round(t_confirm, 2)
        summary["total_s"] = round(t_analyze + t_confirm, 2)
        if r2.status_code != 200:
            print("confirm 失败：", r2.status_code, r2.text[:500])
            return 1

        result = r2.json()
        report = result.get("report") or ""
        summary["report_chars"] = len(report)
        summary["degraded"] = bool(result.get("degraded"))
        summary["sections"] = {
            "质检结果": "## 一、质检结果" in report,
            "差异表达": "## 二、差异表达分析" in report,
            "通路富集": "## 三、通路富集分析" in report,
            "结果解读": "## 四、结果解读" in report,
        }
        summary["cache_hit_after"] = bool(web.CACHE.get(cache_key))

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("\n--- 报告开头 400 字 ---\n" + report[:400])
        return 0
    finally:
        # ── 4. 清理：缓存、pending、历史、临时账号 ─────────────────
        web.CACHE.delete(cache_key)
        if thread_id:
            web.CACHE.delete(f"pending:{thread_id}")
        if not args.user:
            try:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM analysis_history WHERE user_id = :uid"),
                                 {"uid": user_id})
                    conn.execute(text("DELETE FROM users WHERE username = :u"), {"u": username})
            except Exception as exc:  # noqa: BLE001 - 清理失败不影响烟测结论
                print(f"[warn] 清理临时账号失败：{exc}")


if __name__ == "__main__":
    raise SystemExit(main())
