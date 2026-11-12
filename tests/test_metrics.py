"""指标层测试：计数、分位数、计时器、请求 ID 贯穿。"""

import unittest

from omics_agent import metrics


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.m = metrics.Metrics(max_samples=200)

    def test_counter_increments(self):
        self.m.incr("a")
        self.m.incr("a", 2)
        self.assertEqual(self.m.counter("a"), 3)
        self.assertEqual(self.m.counter("missing"), 0)

    def test_percentiles(self):
        for i in range(1, 101):
            self.m.observe("lat", i)
        p = self.m.percentiles("lat")
        # 最近秩取样：p50 落在 50/51 都可接受（round(0.5*99)=50 → 下标 50 的值是 51）
        self.assertIn(p["p50_ms"], (50, 51))
        self.assertEqual(p["p95_ms"], 95)
        self.assertEqual(p["p99_ms"], 99)
        self.assertEqual(p["max_ms"], 100)
        self.assertEqual(p["count"], 100)
        # 分位数必须单调不减，否则报表会自相矛盾
        self.assertLessEqual(p["p50_ms"], p["p90_ms"])
        self.assertLessEqual(p["p90_ms"], p["p95_ms"])
        self.assertLessEqual(p["p95_ms"], p["p99_ms"])

    def test_ring_buffer_is_bounded(self):
        m = metrics.Metrics(max_samples=16)
        for i in range(100):
            m.observe("lat", i)
        self.assertEqual(m.percentiles("lat")["count"], 16)

    def test_timer_records(self):
        with self.m.timer("work_ms"):
            pass
        self.assertEqual(self.m.percentiles("work_ms")["count"], 1)

    def test_timer_records_on_exception(self):
        with self.assertRaises(ValueError):
            with self.m.timer("fail_ms"):
                raise ValueError("x")
        self.assertEqual(self.m.percentiles("fail_ms")["count"], 1)

    def test_snapshot_shape(self):
        self.m.incr("c1")
        with self.m.timer("t1"):
            pass
        snap = self.m.snapshot()
        for field in ("request_id", "uptime_s", "counters", "timers"):
            self.assertIn(field, snap)


class RequestIdTest(unittest.TestCase):
    def test_set_get_reset(self):
        self.assertEqual(metrics.get_request_id(), "-")
        token = metrics.set_request_id("req-123")
        self.assertEqual(metrics.get_request_id(), "req-123")
        metrics.reset_request_id(token)

    def test_new_request_id_is_unique(self):
        self.assertNotEqual(metrics.new_request_id(), metrics.new_request_id())


if __name__ == "__main__":
    unittest.main()
