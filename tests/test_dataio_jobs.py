"""数据层与任务层测试：CSV 解析缓存失效、数据集指纹、异步任务生命周期。"""

import os
import tempfile
import time
import unittest

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
        self.registry = jobs.JobRegistry(max_workers=2, ttl=60, max_jobs=16)

    def test_success_lifecycle(self):
        job_id = self.registry.submit(lambda: 42)
        job = _wait(self.registry, job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"], 42)
        self.assertIsNotNone(job["elapsed_s"])
        self.assertIsNone(job["error"])

    def test_failure_lands_in_job_state(self):
        def boom():
            raise RuntimeError("分析失败")

        job_id = self.registry.submit(boom)
        job = _wait(self.registry, job_id)
        self.assertEqual(job["status"], "error")
        self.assertIn("RuntimeError", job["error"])

    def test_progress_updates_stage(self):
        def work():
            jobs.progress("正在做差异分析")
            return "ok"

        job_id = self.registry.submit(work)
        job = _wait(self.registry, job_id)
        self.assertEqual(job["stage"], "正在做差异分析")

    def test_unknown_job_returns_none(self):
        self.assertIsNone(self.registry.get("no-such-id"))

    def test_stats_counts(self):
        self.registry.submit(lambda: 1)
        time.sleep(0.2)
        self.assertGreaterEqual(self.registry.stats()["jobs_total"], 1)


if __name__ == "__main__":
    unittest.main()
