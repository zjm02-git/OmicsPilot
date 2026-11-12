"""缓存层测试：Redis 主/本地兜底、降级、TTL、key 正确性、single-flight。"""

import os
import tempfile
import threading
import time
import unittest

from omics_agent import cache


class FakeRedis:
    """最小 Redis 替身：够 ReportCache 用的 get/set(nx,ex)/delete/ping。"""

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None, nx=None):
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True

    def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0

    def ping(self):
        return True


class BrokenRedis:
    """模拟 Redis 宕机：所有操作都抛连接错误。"""

    def get(self, *a, **kw):
        raise ConnectionError("redis down")

    def set(self, *a, **kw):
        raise ConnectionError("redis down")

    def delete(self, *a, **kw):
        raise ConnectionError("redis down")

    def ping(self):
        raise ConnectionError("redis down")


class ReportCacheTest(unittest.TestCase):
    def test_redis_roundtrip(self):
        c = cache.ReportCache(redis_client=FakeRedis(), connect=False)
        self.assertTrue(c.set("report:x", "报告正文"))
        value, source = c.get_source("report:x")
        self.assertEqual(value, "报告正文")
        self.assertEqual(source, "redis")

    def test_miss_reports_miss(self):
        c = cache.ReportCache(redis_client=FakeRedis(), connect=False)
        self.assertEqual(c.get_source("nope"), (None, "miss"))

    def test_degrades_to_local_when_redis_dies(self):
        c = cache.ReportCache(redis_client=BrokenRedis(), connect=False)
        self.assertFalse(c.set("report:y", "降级正文"))   # 写 Redis 失败，返回 False
        self.assertEqual(c.get("report:y"), "降级正文")   # 从进程内 LRU 读到了
        self.assertEqual(c.get_source("report:y"), ("降级正文", "local"))

    def test_no_redis_still_works(self):
        c = cache.ReportCache(connect=False, redis_client=None)
        c.set("k", "v")
        self.assertEqual(c.get("k"), "v")

    def test_single_flight_only_one_leader(self):
        c = cache.ReportCache(redis_client=FakeRedis(), connect=False)
        with c.inflight("report:z") as first:
            with c.inflight("report:z") as second:
                self.assertTrue(first)
                self.assertFalse(second)

    def test_lock_released_after_block(self):
        c = cache.ReportCache(redis_client=FakeRedis(), connect=False)
        with c.inflight("report:z"):
            pass
        with c.inflight("report:z") as again:
            self.assertTrue(again)

    def test_wait_for_returns_value_written_by_another_thread(self):
        c = cache.ReportCache(redis_client=FakeRedis(), connect=False)

        def writer():
            time.sleep(0.15)
            c.set("report:w", "别人的结果")

        t = threading.Thread(target=writer)
        t.start()
        value = c.wait_for("report:w", timeout=2.0, interval=0.02)
        t.join()
        self.assertEqual(value, "别人的结果")

    def test_wait_for_timeout_returns_none(self):
        c = cache.ReportCache(redis_client=FakeRedis(), connect=False)
        self.assertIsNone(c.wait_for("report:never", timeout=0.2, interval=0.02))


class TTLTest(unittest.TestCase):
    def test_entry_expires(self):
        c = cache.TTLCache(max_entries=4)
        c.set("a", "v", ttl=1.0)
        self.assertEqual(c.get("a"), "v")
        time.sleep(1.05)
        self.assertIsNone(c.get("a"))

    def test_lru_eviction(self):
        c = cache.TTLCache(max_entries=2)
        c.set("a", 1, ttl=60)
        c.set("b", 2, ttl=60)
        c.set("c", 3, ttl=60)
        self.assertIsNone(c.get("a"))
        self.assertEqual(len(c), 2)


class BuildKeyTest(unittest.TestCase):
    def test_same_input_same_key(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cache.build_key("分析差异", d), cache.build_key("分析差异", d))

    def test_whitespace_normalized(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cache.build_key("分析  差异", d), cache.build_key("分析 差异", d))

    def test_different_dataset_different_key(self):
        """核心回归：同一句需求、不同数据集，必须产出不同的 key。"""
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            with open(os.path.join(d1, "expr.csv"), "w", encoding="utf-8") as f:
                f.write("gene,a,b\ng1,1,2\n")
            with open(os.path.join(d2, "expr.csv"), "w", encoding="utf-8") as f:
                f.write("gene,a,b\ng2,3,4\ng3,5,6\n")
            k1 = cache.build_key("帮我做差异分析", d1)
            k2 = cache.build_key("帮我做差异分析", d2)
            self.assertNotEqual(k1, k2)

    def test_missing_dataset_does_not_crash(self):
        key = cache.build_key("x", os.path.join(tempfile.gettempdir(), "definitely-not-here"))
        self.assertTrue(key.startswith("report:v3:"), key)


if __name__ == "__main__":
    unittest.main()
