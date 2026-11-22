"""数据层与任务层测试：CSV 解析缓存失效、数据集指纹、异步任务生命周期。"""
#docker compose exec worker python tools/demo_failover.py
#脚本会自动走完你要的那套流程


#终端 1 —— 起厨师 + 下单，然后停下来等你：
# docker compose exec worker python tools/demo_failover.py --manual
# 看到打印 [4] 执行中的 worker pid=XXX 就停在这（别关）。
#
# 终端 2 —— 亲手打死那个厨师（把 XXX 换成终端 1 显示的 pid）：
# docker compose exec worker sh -c 'kill -9 XXX'

import os
import tempfile
import threading
import time
import unittest
import uuid

import redis

from omics_agent import dataio, jobs


def _wait(job_registry, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = job_registry.get(job_id)
        if job and job["status"] in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"任务 {job_id} 超时未结束")


class CsvCacheTest(unittest.TestCase):
    def setUp(self):
        dataio.clear_csv_cache()
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "expr.csv")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("gene,s1,s2\ng1,1,2\ng2,3,4\n")

    def test_second_read_hits_cache(self):
        first = dataio.read_csv_cached(self.path)
        second = dataio.read_csv_cached(self.path)
        self.assertIs(first, second)  # 命中缓存返回同一对象，不是重新解析
        self.assertEqual(first.shape, (2, 3))

    def test_invalidated_when_file_changes(self):
        first = dataio.read_csv_cached(self.path)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("gene,s1,s2\ng1,1,2\ng2,3,4\ng3,5,6\n")
        second = dataio.read_csv_cached(self.path)
        self.assertIsNot(first, second)
        self.assertEqual(second.shape, (3, 3))

    def test_cache_info_counts_entries(self):
        dataio.read_csv_cached(self.path)
        self.assertGreaterEqual(dataio.csv_cache_info()["entries"], 1)


class FingerprintTest(unittest.TestCase):
    def test_missing_dir(self):
        self.assertEqual(dataio.dataset_fingerprint(os.path.join(tempfile.gettempdir(), "no-such-dir-xyz")), "missing")

    def test_changes_with_content(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "a.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("x\n1\n")
        first = dataio.dataset_fingerprint(d)
        with open(p, "w", encoding="utf-8") as f:
            f.write("x\n1\n2\n3\n")
        self.assertNotEqual(first, dataio.dataset_fingerprint(d))

    def test_stable_for_unchanged_dir(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "a.csv"), "w", encoding="utf-8") as f:
            f.write("x\n1\n")
        self.assertEqual(dataio.dataset_fingerprint(d), dataio.dataset_fingerprint(d))


class JobRegistryTest(unittest.TestCase):
    def setUp(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
        )
        # 每个用例用独立 stream/group，避免不同用例的 worker 抢同一消费组里的消息
        token = uuid.uuid4().hex[:8]
        self.registry = jobs.JobRegistry(
            redis_client=self.redis,
            ttl=60,
            stream=f"test:jobs:{token}",
            group=f"test:group:{token}",
            dead_stream=f"test:dead:{token}",
        )
        # 注册三种任务：echo 返回参数、boom 抛异常、work 上报阶段
        self.registry.register("echo", lambda payload: payload["value"])
        self.registry.register("boom", lambda payload: (_ for _ in ()).throw(RuntimeError("分析失败")))
        self.registry.register("work", lambda payload: (jobs.progress("正在做差异分析"), "ok")[1])
        # 起一个消费线程，否则 submit 之后没人执行
        self._worker = threading.Thread(target=self.registry.run_worker, daemon=True)
        self._worker.start()

    def tearDown(self):
        try:
            self.redis.delete(self.registry.queue.stream, self.registry.queue.dead_stream)
        except Exception:  # noqa: BLE001
            pass

    def test_success_lifecycle(self):
        job_id = self.registry.submit("echo", {"value": 42})
        job = _wait(self.registry, job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"], 42)
        self.assertNotEqual(job.get("elapsed_s"), "")

    def test_failure_lands_in_job_state(self):
        job_id = self.registry.submit("boom", {})
        job = _wait(self.registry, job_id)
        self.assertEqual(job["status"], "error")
        self.assertIn("RuntimeError", job["error"])

    def test_progress_updates_stage(self):
        job_id = self.registry.submit("work", {})
        job = _wait(self.registry, job_id)
        self.assertEqual(job["stage"], "正在做差异分析")

    def test_unknown_job_returns_none(self):
        self.assertIsNone(self.registry.get("no-such-id"))

    def test_stats_counts(self):
        self.registry.submit("echo", {"value": 1})
        time.sleep(0.5)
        self.assertGreaterEqual(self.registry.stats()["jobs_total"], 1)


if __name__ == "__main__":
    unittest.main()
