"""web.py 集成测试：真 TestClient + 真 Redis + 真 MySQL + 假模型。

为什么需要这一层？（单元测试按定义测不出来）
  单元测试测的是零件：key 怎么拼、熔断第 6 次会不会开、fit_blocks 会不会丢块。
  零件全绿 ≠ 装起来能跑。下面这些 bug 只有"整条链路真跑一遍"才暴露：
    - /analyze 存进 pending:{thread_id} 的 key，/confirm 取出来时格式不一致 → 缓存永远回填不上，且不报错；
    - state 里的 degraded 标记在某一层被吞掉 → 前端把降级结果当正品；
    - multipart 真实解析下，dataset_path 校验/文件名净化不生效；
    - /analyze_async 交给 /jobs/{id} 的字段对不上 → 前端永远看到 queued。

真与假的分界：
  假 —— 只假掉 LLM（FakeModel）：结构化输出直接给 IntentOutput，普通调用返回固定文本。
        真模型一次 ~25s、要 key、要花钱、还会抖，放进单元测试里只会让测试不可靠。
  真 —— Redis、MySQL、pandas 统计、LangGraph 检查点、FastAPI 请求栈，全部是真的。

数据：临时在 data/ 下造 120 基因 × 40 样本（20/20 两组）的小数据集，
      前 12 个基因人为做出 3 倍差异，保证差异分析一定出显著基因、报告一定有四章节。
      用完在 tearDownClass 里删掉，不污染 data/。
"""

from __future__ import annotations

import os

# 限流的桶存在共享 Redis 里，跨运行不清空：反复跑测试会把注册窗口（10 次/小时）
# 耗尽，后续用例全被 429 挡掉，看着像接口坏了。这里关掉限流做测试隔离。
# ratelimit._enabled() 每次调用现读环境变量，因此与 import 顺序无关。
os.environ["RATELIMIT_ENABLED"] = "false"

import os
import shutil
import time
import unittest
import uuid
from unittest import mock

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import text

import web
from db import engine
from omics_agent import dataio, metrics
from omics_agent.cache import build_key
from omics_agent.nodes import IntentOutput

QUESTION = "帮我找两组样本的差异表达基因"
FAKE_REPLY = "[假模型] 先看上调基因，它们提示相关通路被激活。"
PASSWORD = "Passw0rd!itest"


# ══════════ 假模型：只替换 LLM，其余全真 ══════════
class _FakeStructured:
    """顶替 model.with_structured_output(schema) 的返回值。"""

    def __init__(self, intent: str) -> None:
        self._intent = intent

    def invoke(self, _messages):  # noqa: D102 - 签名保持与真实 runnable 一致
        return IntentOutput(analysis_type=self._intent)


class FakeModel:
    """替代 ChatDeepSeek。

    - with_structured_output → 直接返回指定意图（不解析、不联网）
    - invoke → 返回固定文本；fail_invoke=True 时抛错，用于验证"模型挂了仍出报告"
    """

    def __init__(self, intent: str = "differential", reply: str = FAKE_REPLY, fail_invoke: bool = False) -> None:
        self.intent = intent
        self.reply = reply
        self.fail_invoke = fail_invoke
        self.invoke_calls = 0

    def with_structured_output(self, _schema):
        return _FakeStructured(self.intent)

    def invoke(self, _messages):
        self.invoke_calls += 1
        if self.fail_invoke:
            raise RuntimeError("fake gateway 502")
        return AIMessage(content=self.reply)


def _make_dataset(dir_path: str, n_genes: int = 120, n_per_group: int = 20, seed: int = 7) -> None:
    """造一份能被 discover_dataset 认出来的最小数据集。

    识别规则（nodes.discover_dataset）：表达矩阵 = 列数 > 20 且首列唯一、其余列全数字；
    分组表 = 行数恰好等于样本数。所以样本数 40 → 表达矩阵 41 列。
    """
    rng = np.random.default_rng(seed)
    samples = [f"S{i:02d}" for i in range(n_per_group * 2)]
    genes = [f"GENE{i:03d}" for i in range(n_genes)]

    # 值空间必须是 log2（与 fetch_tcga.py / generate_data.py 同一契约）。
    # 线性量纲 100→300 会让 log2fc 变成 200，恰好掩盖了"双重取对数"的口径错误。
    mat = rng.normal(8.0, 0.5, size=(n_genes, len(samples)))
    # 前 12 个基因在 B 组高 3 倍 → log2fc = log2(3) ≈ 1.585 > 1，必然显著
    mat[:12, n_per_group:] += np.log2(3.0)

    expr = pd.DataFrame(mat, columns=samples)
    expr.insert(0, "gene", genes)
    expr.to_csv(os.path.join(dir_path, "expression.csv"), index=False)

    meta = pd.DataFrame(
        {
            # 列名必须是 sample_id：diff.differential_expression 靠它把分组表
            # 接到表达矩阵的列上（这也是 load_qc 现在会校验的契约）
            "sample_id": samples,
            "condition": ["control"] * n_per_group + ["treat"] * n_per_group,
        }
    )
    meta.to_csv(os.path.join(dir_path, "metadata.csv"), index=False)


class WebIntegrationTest(unittest.TestCase):
    """7 个用例，覆盖 /analyze、/confirm、缓存回填、降级不入缓存、异步任务、上传、健康检查。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 数据集必须落在 data/ 之下，否则 /analyze 的路径校验会（正确地）拒绝
        cls.dataset_path = os.path.join(web.BASE_DIR, "data", f"itest_{uuid.uuid4().hex[:8]}")
        os.makedirs(cls.dataset_path, exist_ok=True)
        _make_dataset(cls.dataset_path)

        # 全程替换模型：整个类期间 get_model() 只能拿到假模型。
        # 关键点：/analyze 和 /confirm 是两次独立请求，只在 /analyze 里打补丁的话，
        # 确认阶段恢复执行 interpret_node 时会拿到真模型（既不稳定又要花钱）。
        cls._get_model_patch = mock.patch("omics_agent.nodes.get_model", return_value=FakeModel())
        cls._get_model_mock = cls._get_model_patch.start()

        cls.client = TestClient(web.app)

        # 真注册、真登录、真 MySQL：顺手验证了鉴权链路
        cls.username = f"itest_{uuid.uuid4().hex[:8]}"
        r = cls.client.post("/register", json={"username": cls.username, "password": PASSWORD})
        assert r.status_code == 200, f"注册失败：{r.status_code} {r.text}"
        r = cls.client.post("/login", json={"username": cls.username, "password": PASSWORD})
        assert r.status_code == 200, f"登录失败：{r.status_code} {r.text}"
        cls.headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        cls.user_id = cls.client.get("/me", headers=cls.headers).json()["user_id"]
        cls.created_threads: list[str] = []

    @classmethod
    def tearDownClass(cls) -> None:
        cls._get_model_patch.stop()
        # 清缓存（含 pending）与临时数据集，别给下一次运行留脏状态
        for tid in cls.created_threads:
            web.CACHE.delete(f"pending:{tid}")
        web.CACHE.delete(build_key(QUESTION, cls.dataset_path, user_id=cls.user_id))
        shutil.rmtree(cls.dataset_path, ignore_errors=True)
        dataio.clear_csv_cache()

        # 清 MySQL 里的测试账号与历史（历史有外键，先删子表）
        try:
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM analysis_history WHERE user_id = :uid"), {"uid": cls.user_id})
                conn.execute(text("DELETE FROM users WHERE username = :u"), {"u": cls.username})
        except Exception:  # noqa: BLE001 - 清理失败不该让测试变红
            pass

    # ── 辅助 ──────────────────────────────────────────────
    def _use_model(self, model: FakeModel) -> FakeModel:
        """切换当前生效的假模型（补丁本身在 setUpClass 就打开了，覆盖整个测试类）。"""
        type(self)._get_model_mock.return_value = model
        return model

    def _key(self) -> str:
        # 缓存 key 现在含用户维度：必须用当前登录用户去算，否则删不到/查不到自己的键
        return build_key(QUESTION, self.dataset_path, user_id=self.user_id)

    def _counter(self, name: str) -> int:
        snap = metrics.snapshot()
        counters = snap.get("counters") if isinstance(snap.get("counters"), dict) else snap
        return int((counters or {}).get(name, 0))

    def _payload(self, dataset_path: str | None = None) -> dict:
        return {
            "user_request": QUESTION,
            "dataset_path": dataset_path or self.dataset_path,
            "user_id": self.username,
        }

    def _analyze(self, model: FakeModel | None = None) -> tuple[dict, str]:
        """跑一次 /analyze（冷启动），返回 (响应体, 缓存 key)。"""
        model = model or FakeModel()
        key = self._key()
        web.CACHE.delete(key)  # 每个用例都从"缓存未命中"开始，否则测不到要测的分支
        self._use_model(model)
        r = self.client.post("/analyze", json=self._payload(), headers=self.headers)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["status"], "pending_confirm", body)
        self.created_threads.append(body["thread_id"])
        return body, key

    def _confirm(self, thread_id: str, approved: bool = True) -> dict:
        r = self.client.post(
            "/confirm", json={"thread_id": thread_id, "approved": approved}, headers=self.headers
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def _poll(self, job_id: str, timeout: float = 60.0) -> dict:
        deadline = time.monotonic() + timeout
        last: dict = {}
        while time.monotonic() < deadline:
            last = self.client.get(f"/jobs/{job_id}", headers=self.headers).json()
            if last.get("status") in ("done", "error"):
                return last
            time.sleep(0.05)
        self.fail(f"任务 {job_id} 在 {timeout}s 内没跑完，最后状态：{last}")

    # ── 1. /analyze 挂起 + 记下缓存 key ───────────────────
    def test_01_analyze_pends_and_remembers_cache_key(self) -> None:
        body, key = self._analyze()

        self.assertTrue(body["thread_id"])
        self.assertIn("质检通过", body["qc_report"])
        self.assertEqual(body["group_a"], "control")
        self.assertEqual(body["group_b"], "treat")

        # 关键接线：/analyze 记下的 key 必须与 /confirm 回填时用的完全一致，
        # 否则报告算完了却永远进不了缓存，而且没有任何报错。
        self.assertEqual(web.CACHE.get(f"pending:{body['thread_id']}"), key)

    # ── 2. /confirm 出报告 + 回填缓存 + 二次命中 ──────────
    def test_02_confirm_reports_and_backfills_cache(self) -> None:
        body, key = self._analyze()
        self.assertIsNone(web.CACHE.get(key), "挂起阶段不该有完整报告进缓存")

        result = self._confirm(body["thread_id"])
        report = result["report"]
        self.assertIn("# 多组学数据分析报告", report)
        self.assertIn("## 四、结果解读", report)
        self.assertIn(FAKE_REPLY, report, "假模型必须真的参与了报告生成")
        self.assertNotIn("degraded", result)

        # 回填：这是"差异分析主路径永远命中不了缓存"的修复点
        self.assertEqual(web.CACHE.get(key), report)

        # 同一问题再问一次 → 应命中缓存，且报告逐字一致
        self._use_model(FakeModel())
        r2 = self.client.post("/analyze", json=self._payload(), headers=self.headers)
        self.assertEqual(r2.status_code, 200, r2.text)
        cached = r2.json()
        self.assertTrue(cached.get("from_cache"), cached)
        self.assertIn(cached.get("cache_source"), ("redis", "local"))
        self.assertEqual(cached["report"], report)
        self.assertGreaterEqual(self._counter("cache_backfill_from_confirm_total"), 1)

    # ── 3. 降级报告绝不入缓存 ─────────────────────────────
    def test_03_degraded_report_is_not_cached(self) -> None:
        before = self._counter("cache_skip_degraded_total")
        body, key = self._analyze(model=FakeModel(fail_invoke=True))

        result = self._confirm(body["thread_id"])
        # 模型挂了也必须给报告（只剩统计摘要），但要明确告诉调用方
        self.assertTrue(result.get("degraded"), result)
        self.assertIn("interpret", result.get("degraded_stages", []))
        self.assertIn("# 多组学数据分析报告", result["report"])

        # 这是本次最重要的正确性约束：一次模型抖动不能污染接下来 24 小时的所有同问请求
        self.assertIsNone(web.CACHE.get(key), "降级报告被写进缓存了")
        self.assertEqual(self._counter("cache_skip_degraded_total"), before + 1)

    # ── 4. 异步任务生命周期 ───────────────────────────────
    def test_04_async_job_lifecycle(self) -> None:
        web.CACHE.delete(self._key())
        self._use_model(FakeModel())
        r = self.client.post("/analyze_async", json=self._payload(), headers=self.headers)
        if os.getenv("USE_POSTGRES", "").lower() != "true":
            # 没有持久化检查点时必须明确拒绝：否则用户会拿到一个永远 confirm 不了的 job
            self.assertEqual(r.status_code, 503, r.text)
            self.assertIn("USE_POSTGRES", r.text)
            return
        self.assertEqual(r.status_code, 200, r.text)
        submitted = r.json()
        self.assertEqual(submitted["status"], "queued")
        self.assertTrue(submitted["job_id"])

        # 提交后立刻返回，任务在线程里跑；这里等它结束
        job = self._poll(submitted["job_id"])

        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["result"]["status"], "pending_confirm")
        self.assertIn("质检通过", job["result"]["qc_report"])
        self.created_threads.append(job["result"]["thread_id"])

        # 不存在的任务要 404，而不是 500 或空对象
        self.assertEqual(self.client.get("/jobs/nope", headers=self.headers).status_code, 404)

    # ── 5. 真 multipart 上传 → 用上传目录分析 ─────────────
    def test_05_upload_then_analyze_uploaded_files(self) -> None:
        with open(os.path.join(self.dataset_path, "expression.csv"), "rb") as f:
            expr_bytes = f.read()
        with open(os.path.join(self.dataset_path, "metadata.csv"), "rb") as f:
            meta_bytes = f.read()

        r = self.client.post(
            "/upload",
            files=[
                ("files", ("expression.csv", expr_bytes, "text/csv")),
                ("files", ("metadata.csv", meta_bytes, "text/csv")),
            ],
            headers=self.headers,
        )
        self.assertEqual(r.status_code, 200, r.text)
        dataset_path = r.json()["dataset_path"]  # 绝对路径，落在 UPLOAD_ROOT 下
        try:
            # 上传必须落在 UPLOAD_ROOT 里。以前这里断言 startswith("data/upload")，
            # 那断的是"相对 CWD 落盘"的旧契约 —— 而相对落盘会在换工作目录启动时
            # 造成"上传成功、分析一律 400（不在 data/ 内）"，是个真缺陷。
            self.assertEqual(
                os.path.realpath(os.path.dirname(dataset_path)), web.UPLOAD_ROOT
            )
            self.assertTrue(os.path.isdir(dataset_path))

            # 真上传目录走完整分析：证明"文件名净化 + 路径校验 + 内容识别"三者能串起来
            self._use_model(FakeModel())
            r2 = self.client.post(
                "/analyze",
                json=self._payload(os.path.abspath(os.path.join(web.BASE_DIR, dataset_path))),
                headers=self.headers,
            )
            self.assertEqual(r2.status_code, 200, r2.text)
            body = r2.json()
            self.assertEqual(body["status"], "pending_confirm", body)
            self.assertIn("20 样本", body["qc_report"])  # 识别出的是上传的那份数据集
            self.created_threads.append(body["thread_id"])
        finally:
            shutil.rmtree(os.path.join(web.BASE_DIR, dataset_path), ignore_errors=True)

    # ── 6. 健康检查 + 指标 ────────────────────────────────
    def test_06_healthz_and_metrics(self) -> None:
        r = self.client.get("/healthz")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["checks"]["mysql"], f"MySQL 不可用：{body}")
        self.assertIn(body["status"], ("ok", "degraded"))
        self.assertIn("cache", body)
        self.assertIn("jobs", body)

        r2 = self.client.get("/metrics", headers=self.headers)
        self.assertEqual(r2.status_code, 200, r2.text)
        snap = r2.json()
        self.assertIn("counters", snap)
        # 前面几个用例真的跑过 /analyze，指标里必须留下痕迹
        self.assertGreaterEqual(int(snap["counters"].get("cache_miss_total", 0)), 1)

    # ── 7. 路径穿越与鉴权边界 ─────────────────────────────
    def test_07_rejects_path_outside_data_and_missing_token(self) -> None:
        r = self.client.post(
            "/analyze",
            json={"user_request": QUESTION, "dataset_path": "C:\\Windows"},
            headers=self.headers,
        )
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("data", r.json()["detail"])

        r2 = self.client.post("/analyze", json=self._payload())
        self.assertEqual(r2.status_code, 401, r2.text)


    # ── 8. 质检必须拦住"样本对不上"的分组表 ────────────────
    def test_08_qc_rejects_metadata_without_sample_id(self) -> None:
        """这条用例是 test_02 暴露出来的真 bug 的回归测试。

        原状：metadata 缺 sample_id 时质检照样通过，用户点"确认"后才在 diff 里炸，
        报的是"会话可能已失效"——与真实原因（列名不对）毫无关系。
        """
        bad_dir = os.path.join(web.BASE_DIR, "data", f"itest_bad_{uuid.uuid4().hex[:6]}")
        os.makedirs(bad_dir, exist_ok=True)
        try:
            _make_dataset(bad_dir)
            meta_path = os.path.join(bad_dir, "metadata.csv")
            df = pd.read_csv(meta_path)
            df.rename(columns={"sample_id": "sample"}, inplace=True)
            df.to_csv(meta_path, index=False)

            self._use_model(FakeModel())
            r = self.client.post("/analyze", json=self._payload(bad_dir), headers=self.headers)

            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            # 质检失败不挂起：直接出"质检未通过"的报告，不浪费用户一次点击
            self.assertEqual(body["status"], "done", body)
            self.assertIn("质检未通过", body["report"])
            self.assertIn("sample_id", body["report"])
        finally:
            shutil.rmtree(bad_dir, ignore_errors=True)

    # ── 9. 阈值可自定义 + 越界必须被后端拒绝 ────────────────
    def test_09_threshold_bounds_and_validation(self) -> None:
        """HITL 阈值从「只有三档」改成「档位 + 自定义数值」。

        守三条：
          1. /analyze 必须把 threshold_bounds 抛给前端，否则输入框没有 min/max，
             前端只能退回硬编码默认值 —— 等于后端改了前端不知道；
          2. 合法自定义值（文献常见的 |log2FC|>0.8、FDR<0.02）必须能通过；
          3. 越界值必须 422，且**不消耗这次会话**：用户改对之后还能继续确认。
             如果 422 把 interrupt 吃掉或把 state 弄脏，用户只能重跑一遍分析。
        """
        body, _key = self._analyze()
        thread_id = body["thread_id"]

        # 1. 前端渲染输入框所需的边界必须透出
        bounds = body.get("threshold_bounds")
        self.assertIsInstance(bounds, dict, body)
        for key in ("fc_threshold", "fdr_threshold"):
            self.assertIn(key, bounds)
            self.assertLess(bounds[key]["min"], bounds[key]["max"])
        self.assertIn("current_thresholds", body)
        self.assertIn("threshold_options", body)

        # 2a. 越界 |log2FC| → 422，且不进图（不会产生报告）
        before = self._counter("threshold_invalid_rejected_total")
        bad = self.client.post(
            "/confirm",
            json={"thread_id": thread_id, "approved": True, "fc_threshold": 9.0},
            headers=self.headers,
        )
        self.assertEqual(bad.status_code, 422, bad.text)
        self.assertIn("log2FC", bad.json()["detail"])
        self.assertIn("5", bad.json()["detail"])  # 文案里必须带上允许范围
        self.assertGreaterEqual(self._counter("threshold_invalid_rejected_total"), before + 1)

        # 2b. 越界 FDR → 422
        bad2 = self.client.post(
            "/confirm",
            json={"thread_id": thread_id, "approved": True, "fdr_threshold": 0.9},
            headers=self.headers,
        )
        self.assertEqual(bad2.status_code, 422, bad2.text)
        self.assertIn("FDR", bad2.json()["detail"])

        # 3. 会话没被越界请求弄脏：改对之后仍能正常出报告
        ok = self.client.post(
            "/confirm",
            json={
                "thread_id": thread_id,
                "approved": True,
                "fc_threshold": 0.8,
                "fdr_threshold": 0.02,
            },
            headers=self.headers,
        )
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertIn("# 多组学数据分析报告", ok.json()["report"])


if __name__ == "__main__":
    unittest.main()
