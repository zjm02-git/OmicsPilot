"""韧性层测试：重试/退避、熔断、结构化输出自纠错与兜底、关键词路由。"""

import unittest

from omics_agent import metrics, resilience


class FakeStructured:
    def __init__(self, owner):
        self.owner = owner

    def invoke(self, messages):
        self.owner.calls.append(list(messages))
        if len(self.owner.calls) <= self.owner.fail_times:
            raise ValueError("结构化输出解析失败")
        return "parsed"


class FakeModel:
    """可控失败次数的模型替身：记录每次拿到的 messages。"""

    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls = []

    def with_structured_output(self, schema):
        return FakeStructured(self)


class CallWithRetryTest(unittest.TestCase):
    def test_succeeds_on_third_attempt(self):
        slept = []
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise RuntimeError("boom")
            return "ok"

        out = resilience.call_with_retry(flaky, attempts=3, sleep=slept.append, label="t_retry")
        self.assertEqual(out, "ok")
        self.assertEqual(state["n"], 3)
        self.assertEqual(len(slept), 2)
        # 指数退避：第二次等待应明显长于第一次（含 ±30% 抖动，仍单调）
        self.assertGreater(slept[1], slept[0])

    def test_exhausted_raises_original_error(self):
        state = {"n": 0}

        def always_fail():
            state["n"] += 1
            raise ValueError("永久失败")

        with self.assertRaises(ValueError):
            resilience.call_with_retry(always_fail, attempts=2, sleep=lambda s: None, label="t_fail")
        self.assertEqual(state["n"], 2)

    def test_breaker_opens_after_threshold(self):
        breaker = resilience.CircuitBreaker("t", failure_threshold=2, reset_timeout=60)
        state = {"n": 0}

        def always_fail():
            state["n"] += 1
            raise RuntimeError("网关超时")

        with self.assertRaises(resilience.CircuitOpenError):
            resilience.call_with_retry(always_fail, attempts=5, breaker=breaker,
                                       sleep=lambda s: None, label="t_breaker")
        self.assertEqual(breaker.state, "open")
        self.assertFalse(breaker.allow())
        self.assertEqual(state["n"], 2)  # 熔断后不再真正调用

    def test_breaker_recovers_on_success(self):
        breaker = resilience.CircuitBreaker("ok", failure_threshold=1, reset_timeout=0.0)
        with self.assertRaises(RuntimeError):
            resilience.call_with_retry(lambda: (_ for _ in ()).throw(RuntimeError("x")),
                                       attempts=1, breaker=breaker, label="t_rec")
        self.assertEqual(breaker.state, "open")
        out = resilience.call_with_retry(lambda: "done", attempts=1, breaker=breaker, label="t_rec")
        self.assertEqual(out, "done")
        self.assertEqual(breaker.state, "closed")


class StructuredOutputTest(unittest.TestCase):
    def test_self_correction_on_second_attempt(self):
        model = FakeModel(fail_times=1)
        out = resilience.structured_output_call(model, schema=dict, messages=["请分析这段需求"],
                                                sleep=lambda s: None)
        self.assertEqual(out, "parsed")
        self.assertEqual(len(model.calls), 2)
        # 第二次比第一次多出一条纠错提示
        self.assertEqual(len(model.calls[1]), len(model.calls[0]) + 1)

    def test_fallback_used_after_all_attempts(self):
        model = FakeModel(fail_times=99)
        seen = {}

        def fallback(exc):
            seen["exc"] = exc
            return "fallback-value"

        out = resilience.structured_output_call(model, schema=dict, messages=["x"], attempts=2,
                                                fallback=fallback, sleep=lambda s: None)
        self.assertEqual(out, "fallback-value")
        self.assertIsInstance(seen["exc"], Exception)
        self.assertEqual(len(model.calls), 2)

    def test_without_fallback_raises(self):
        model = FakeModel(fail_times=99)
        with self.assertRaises(ValueError):
            resilience.structured_output_call(model, schema=dict, messages=["x"], attempts=2,
                                              sleep=lambda s: None)


class KeywordIntentTest(unittest.TestCase):
    def test_routes(self):
        cases = {
            "帮我做差异分析看哪些基因上调": "differential",
            "做个 PCA 降维看看样本分布": "pca",
            "用聚类把样本分群": "clustering",
            "TP53 这个基因的功能是什么": "qa",
            "今天天气不错": "other",
        }
        for text, expected in cases.items():
            self.assertEqual(resilience.keyword_intent(text), expected, text)

    def test_empty_text(self):
        self.assertEqual(resilience.keyword_intent(""), "other")


class TimeoutTest(unittest.TestCase):
    def test_timeout_raises(self):
        class SlowModel:
            def invoke(self, messages):
                import time
                time.sleep(0.5)
                return "late"

        before = metrics.METRICS.counter("llm_timeout_total")
        with self.assertRaises(TimeoutError):
            resilience.invoke_with_timeout(SlowModel(), ["hi"], timeout_s=0.05)
        self.assertGreater(metrics.METRICS.counter("llm_timeout_total"), before)


if __name__ == "__main__":
    unittest.main()
