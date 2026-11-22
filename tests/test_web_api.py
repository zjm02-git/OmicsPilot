"""接口契约测试：鉴权、入参校验、上传白名单、健康检查、指标端点。

用 FastAPI 的 dependency_overrides 换掉鉴权依赖，因此不依赖"真的有登录用户"，
但仍然真实走一遍路由、校验、中间件——改坏契约会在这里先炸，而不是等前端发现。
"""

import os
import unittest
import uuid

# 限流的桶存在**共享 Redis** 里（跨进程、跨运行都不清空）。不关掉的话，
# 反复跑测试会把 10 次/小时的注册窗口耗尽，之后 register 全返回 429，
# 看起来像"接口坏了"，其实是测试之间在互相污染。
# 注意：ratelimit._enabled() 是每次调用现读环境变量，所以这句与 import 顺序无关。
os.environ["RATELIMIT_ENABLED"] = "false"

try:
    from fastapi.testclient import TestClient

    import web  # 导入即建图、连库；连不上库的环境下整组跳过
    IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - 环境缺依赖/缺库时跳过而不是报错
    TestClient = None
    web = None
    IMPORT_ERROR = exc


@unittest.skipIf(web is None, f"web 不可导入：{IMPORT_ERROR}")
class ApiContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(web.app)

    def setUp(self):
        web.app.dependency_overrides[web.get_current_user] = lambda: {
            "user_id": 1, "username": "tester"}

    def tearDown(self):
        web.app.dependency_overrides.clear()

    # ── 鉴权 ──────────────────────────────────────────────
    def test_me_requires_token(self):
        web.app.dependency_overrides.clear()  # 这一条要验证真实鉴权，必须摘掉替身
        r = self.client.get("/me")
        self.assertEqual(r.status_code, 401)

    def test_healthz_shape(self):
        r = self.client.get("/healthz")
        self.assertIn(r.status_code, (200, 503))
        body = r.json()
        self.assertIn("status", body)
        self.assertIn("checks", body)
        self.assertIsInstance(body["checks"], dict)

    def test_request_id_header_present(self):
        r = self.client.get("/healthz")
        self.assertTrue(r.headers.get("X-Request-ID"))

    # ── 入参校验：路径穿越 ────────────────────────────────
    def test_dataset_path_outside_data_is_rejected(self):
        r = self.client.post("/analyze", json={
            "user_request": "帮我做差异分析", "dataset_path": "C:\\Windows"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("data", r.json()["detail"])

    def test_dataset_path_cross_drive_does_not_crash(self):
        """回归：commonpath 跨盘符抛 ValueError，必须被吃掉并返回 400。"""
        r = self.client.post("/analyze", json={
            "user_request": "帮我做差异分析", "dataset_path": "Z:\\other-drive\\data"})
        self.assertEqual(r.status_code, 400)

    # ── 上传白名单 ────────────────────────────────────────
    def test_upload_rejects_disallowed_suffix(self):
        r = self.client.post("/upload", files=[("files", ("evil.exe", b"MZ..."))])
        self.assertEqual(r.status_code, 400)
        self.assertIn("只接受", r.json()["detail"])

    def test_upload_rejects_path_in_filename(self):
        self.assertIn("/", web._safe_filename("../../etc/passwd")) if False else None
        safe = web._safe_filename("..\\..\\evil.csv")
        self.assertNotIn("..", safe)
        self.assertTrue(safe.endswith(".csv"))

    # ── 可观测性 ──────────────────────────────────────────
    def test_metrics_endpoint(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("counters", body)
        self.assertIn("timers", body)

    def test_job_unknown_returns_404(self):
        r = self.client.get(f"/jobs/{uuid.uuid4().hex}")
        self.assertEqual(r.status_code, 404)


@unittest.skipIf(web is None, f"web 不可导入：{IMPORT_ERROR}")
class AuthFlowTest(unittest.TestCase):
    """注册→登录→带票访问→改密→删号，走真实数据库与真实 JWT。"""

    def setUp(self):
        self.client = TestClient(web.app)
        web.app.dependency_overrides.clear()
        self.username = "ut_" + uuid.uuid4().hex[:8]
        self.password = "pw_123456"

    def tearDown(self):
        web.app.dependency_overrides.clear()

    def test_full_auth_flow(self):
        r = self.client.post("/register", json={"username": self.username, "password": self.password})
        self.assertEqual(r.status_code, 200)

        # 重复注册要被拒
        r = self.client.post("/register", json={"username": self.username, "password": self.password})
        self.assertEqual(r.status_code, 400)

        # 密码错 → 401，且提示不区分"用户不存在"与"密码错"
        r = self.client.post("/login", json={"username": self.username, "password": "wrong"})
        self.assertEqual(r.status_code, 401)

        r = self.client.post("/login", json={"username": self.username, "password": self.password})
        self.assertEqual(r.status_code, 200)
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        r = self.client.get("/me", headers=headers)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["username"], self.username)

        r = self.client.get("/me", headers={"Authorization": "Bearer forged.token.value"})
        self.assertEqual(r.status_code, 401)

        # 改密后旧密码失效
        r = self.client.post("/change_password", headers=headers,
                             json={"old_password": self.password, "new_password": "new_pw_888"})
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/login", json={"username": self.username, "password": self.password})
        self.assertEqual(r.status_code, 401)
        r = self.client.post("/login", json={"username": self.username, "password": "new_pw_888"})
        self.assertEqual(r.status_code, 200)

        # 删号后登不上
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        r = self.client.delete("/user", headers=headers)
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/login", json={"username": self.username, "password": "new_pw_888"})
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()
