# 多组学数据分析智能体（OmicsAgent / GenPilot）

> 用 **LangGraph** 把「基因表达数据分析」这条多步骤流程，编排成一个**能持久化、能人工确认、能记住偏好、能自动解读、并且扛得住外部依赖故障**的 AI Agent。

[![tests](https://img.shields.io/badge/tests-80%20passed-brightgreen)]() [![coverage](https://img.shields.io/badge/coverage-74.8%25-green)]() [![python](https://img.shields.io/badge/python-3.12-blue)]()

---

## 一、这个项目是干嘛的

生物数据分析（比如差异表达分析）不是"问一句答一句"，而是一条**流水线**：

```
拿到表达数据 → 质检 → 差异分析 → 通路富集 → 解读结果 → 出报告
```

这条流水线有三大痛点：

1. **步骤多、容易断** —— 跑到一半出错，得从头再来；
2. **参数多、要试** —— fold change 阈值、p 值阈值选多少，不同选择结果不同；
3. **结果要解读** —— 一堆基因列表，得有人看懂它有什么生物学意义。

**OmicsAgent 做的事**：用 LangGraph 的「图」把这条流水线变成一个个**节点**，让大模型（LLM）负责"理解需求 + 解读结果"，让 Python 负责"质检 + 统计计算"，中间用**检查点**保存进度、用**条件边**做路由。这样一个不懂生物的人，说一句"帮我找差异表达基因"，就能拿到一份带解读的分析报告。

**但它不只是一个能跑的 demo**：真实模型一次分析要 25 秒上下，外部依赖（Redis / 模型网关 / MySQL）任何一环抖动都会让请求失败。所以外面还套了一层**服务层**——缓存一致性、降级、熔断重试、上下文预算、异步任务、健康检查与指标、80 个自动化测试。这一层才是它像"系统"而不是"脚本"的地方。

---

## 二、核心亮点

### 2.1 Agent / 编排层

| 亮点 | 用在哪 | 解决什么 |
| --- | --- | --- |
| **StateGraph 图编排** | 节点串成流水线 | 把多步骤流程显式画成图，可读可维护 |
| **结构化输出路由** | 解析用户需求，判断分析类型 | 让 LLM 输出"确定的值"而不是自由文本，路由才可靠 |
| **条件边** | 质检不合格 → 走"失败报告"分支 | 流程能自己判断该走哪条路 |
| **Checkpointer 持久化** | InMemorySaver / PostgresSaver 可切换 | 检查点存库，中断接着跑、重启不丢 |
| **Human-in-the-loop** | 质检后 interrupt 暂停等确认 | 关键步骤人工把关，结果可审核 |
| **长期记忆** | InMemoryStore / PostgresStore | 记住用户上次的阈值，下次自动复用 |
| **真实统计计算** | t 检验 + BH 校正 + log2FC | 差异分析是真的在算，不是 LLM 瞎编 |
| **通路富集分析** | 超几何检验 + BH 校正 | 找出差异基因扎堆的功能通路 |
| **LLM 结果解读** | 把基因列表翻译成人话 | 结果可解释 |
| **回溯 / 时间旅行** | get_state_history + update_state | 跑完回到历史检查点换参数重跑 |

### 2.2 服务层（本轮工程化加固）

| 模块 | 解决的问题 | 关键做法 |
| --- | --- | --- |
| `omics_agent/cache.py` | **缓存串数据**（换数据集问同一句返回旧报告）、Redis 单点、缓存击穿 | key 含数据集指纹；Redis + 进程内 LRU 双写降级；TTL ±10% 抖动；`SET NX` single-flight |
| `omics_agent/resilience.py` | 模型网关超时 / 502 / 限流 / 结构化输出解析失败 | 线程池硬超时；指数退避 + 抖动重试；三态熔断；自纠错重试 → 关键词规则兜底 |
| `omics_agent/context.py` | prompt 悄悄顶满上下文、被截断的恰是最重要证据 | 按 priority 装 token 预算；被裁块在 prompt 中显式声明 |
| `omics_agent/dataio.py` | 37MB CSV 一轮分析被解析 3~4 次 | 按 `(mtime, size, kwargs)` 签名缓存解析结果；数据集指纹 |
| `omics_agent/jobs.py` + `queue.py` | 25s 长任务占死 HTTP 连接 | Redis Stream 真队列（XADD→XREADGROUP→XACK→XAUTOCLAIM→死信），kill -9 不丢任务 |
| `omics_agent/metrics.py` | 出问题只能猜"哪慢、命中没、是哪个请求" | 零依赖计数器 + 环形采样分位数 + `contextvars` 请求 ID |
| `web.py`（重写） | 每个上层能力都要在 HTTP 层落地 | 幂等缓存键、single-flight、`/healthz`、`/metrics`、`/analyze_async`、路径穿越校验、降级结果不入缓存 |

---

## 三、架构图

```
用户："帮我找两组样本的差异表达基因，并解读"
        │
        ▼  POST /analyze  （或 /analyze_async → GET /jobs/{id}）
┌──────────────────────────────────────────────────────────┐
│ 服务层  observability 中间件（X-Request-ID / 耗时 / 状态码）  │
│  ├─ cache.get(key)            key = 问题 + 路径 + 数据集指纹 │
│  ├─ single-flight(SET NX)     并发同问只跑一次图             │
│  └─ Redis 挂 → 进程内 LRU 兜底（仍 200）                    │
└──────────────────────────────────────────────────────────┘
        │
        ▼  graph.invoke()
┌──────────────────────────────────────────────────────────┐
│ State（状态）：一条分析任务从头到尾共享的数据                │
└──────────────────────────────────────────────────────────┘
        │
        ▼
[① parse_intent]   结构化输出 → 判断分析类型（失败 → 关键词兜底）
        │
        ├── other ─────────► [⑧ report] 出"暂不支持"报告
        ▼ differential
[② load_qc]        加载数据（CSV 解析缓存）+ 质检 + 样本对齐校验
        │
        ├── 质检不合格 ──► [⑧ report] 出"质检失败报告"
        ▼ 质检合格
[③ human_review]   interrupt 暂停，等人工确认
        │
        ├── 用户取消 ──► [⑧ report] 出"已取消"报告
        ▼ 用户批准（POST /confirm 续跑）
[④ diff_analysis]  差异表达分析（t 检验 + BH 校正，numpy 向量化）
        │
        ▼
[⑤ enrichment]     通路富集分析（超几何检验 + BH 校正）
        │
        ▼
[⑥ interpret]      上下文预算裁证据 → LLM 解读（降级为统计摘要）
        │
        ▼
[⑦ save_memory]    把本次阈值偏好写回长期记忆
        │
        ▼
[⑧ report]         汇总成 Markdown 报告 → 回填缓存（降级结果不回填）
        │
        ▼
       END
```

---

## 四、目录结构

```
多组学数据分析智能体/
├── web.py                   # FastAPI 服务层：鉴权 / 上传 / 分析 / 异步任务 / 健康检查 / 指标
├── db.py                    # SQLAlchemy 模型（users / analysis_history）
├── app.py                   # CLI 入口：跑一次完整分析（含回溯重跑）
├── backtrack_demo.py        # 回溯（时间旅行）演示
├── worker.py               # 独立 worker 进程：消费 Redis Stream 任务
├── omics_agent/             # 核心包
│   ├── config.py            # 读配置、造模型、造检查点/记忆（工厂模式）
│   ├── state.py             # State 定义（含 degraded 降级标记）
│   ├── graph.py             # 节点拼图 + 条件路由 + Checkpointer/Store
│   ├── nodes.py             # 各节点函数（质检 / 分析 / 解读 / 报告）
│   ├── diff.py              # 差异表达分析（numpy 向量化 Welch t + BH）
│   ├── enrichment.py        # 通路富集（超几何检验 + BH）
│   ├── rag.py               # 基因卡片 RAG（bge-m3 + Chroma 双检索）
│   ├── cache.py             # 缓存层：Redis + 本地兜底、TTL 抖动、single-flight
│   ├── resilience.py        # 韧性层：超时 / 重试 / 熔断 / 结构化输出兜底
│   ├── context.py           # 上下文预算：按优先级裁剪证据 + 裁剪声明
│   ├── dataio.py            # CSV 解析缓存 + 数据集指纹
│   ├── jobs.py              # 异步任务注册表（状态存 Redis Hash，队列走 Stream）
│   ├── queue.py             # Redis Stream 队列（XADD/XREADGROUP/XAUTOCLAIM/死信）
│   └── metrics.py           # 零依赖指标：计数器 + P50/P95/P99 + 请求 ID
├── frontend/app.py          # Streamlit 前端
├── tests/                   # 80 个 unittest 用例（含 8 个 Web 集成测试）
├── tools/
│   ├── coverage_report.py   # 自建覆盖率工具（compile + settrace，无需 coverage 包）
│   ├── bench.py             # 压测 / 故障注入（缓存命中、冷路径、Redis 挂、LLM 挂、重试 A/B）
│   ├── e2e_live.py          # 真模型端到端烟测（默认不进回归）
│   └── demo_failover.py     # 容错演示：kill -9 worker 任务被接走
├── bench/last_run.json      # 最近一次压测结果（可复核）
├── cov.json                 # 最近一次覆盖率结果
├── data/                    # 数据（fetch_tcga.py / fetch_go.py / generate_data.py）
├── docs/                    # 项目文档 + 工程化升级说明
├── docker-compose.yml       # mysql / postgres / redis / backend / worker / frontend
├── requirements.txt
└── .env.example
```

---

## 五、接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/register` `/login` | 账号（JWT + bcrypt） |
| GET | `/me` · DELETE `/user` · POST `/change_password` | 账号管理 |
| POST | `/upload` | 上传数据集（白名单 + 200MB 上限 + 落在 `data/` 内） |
| POST | `/analyze` | 同步分析，返回 `pending_confirm`（含质检摘要）或 `done` |
| POST | `/confirm` | HITL 续跑（`thread_id` 关联检查点，轮数封顶 5） |
| POST | `/analyze_async` · GET `/jobs/{id}` | 异步长任务 + 轮询（阶段进度 / 错误 / 结果） |
| GET | `/history` | 历史记录（pending→done） |
| GET | `/healthz` | 依赖探活（mysql / redis / checkpointer），区分 `ok` / `degraded` / `down` |
| GET | `/metrics` | 计数器 + 各路由 P50/P90/P95/P99（需鉴权） |

---

## 六、快速开始

### 1. 装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型 Key

把 `.env.example` 复制成 `.env`，填上 DeepSeek Key：

```
DEEPSEEK_API_KEY=你的key
REDIS_HOST=127.0.0.1
SECRET_KEY=随便一串足够长的随机串
```

### 3. 准备数据（二选一）

```bash
python data/generate_data.py     # 模拟数据：100 基因 × 8 样本，开箱即用
# 或
python data/fetch_tcga.py        # 真实 TCGA-BRCA：226 样本 × 约 2 万基因
python data/fetch_go.py          # GO 注释（MSigDB）
```

两种方式生成相同的三个 csv（`expression.csv` / `metadata.csv` / `gene_ontology.csv`），代码不用改。

### 4. 起服务

```bash
uvicorn web:app --reload --port 8000     # 后端
streamlit run frontend/app.py            # 前端
# 或
docker compose up -d                     # 五服务一键起
```

### 5.（可选）启用 PostgreSQL 持久化

在 `.env` 里设 `USE_POSTGRES=true`，检查点与长期记忆就存进 PostgreSQL（重启不丢）。默认内存版开箱即跑，无需数据库。

---

## 七、验证：测试 / 覆盖率 / 压测

```bash
python -m unittest discover -s tests -t .        # 80 个用例
python tools/coverage_report.py --json cov.json  # 自建覆盖率工具
python tools/bench.py --port 8011                # 压测 + 故障注入（需先起服务）
python tools/e2e_live.py                         # 真模型端到端（会消耗 token）
```

**最近一次实测**（`bench/last_run.json`、`cov.json`）：

| 项目 | 结果 |
| --- | --- |
| 单元 + 集成测试 | **80 passed / 0 failed** |
| 覆盖率 | **74.8%**（1480/1979 行；web.py 67.5%） |
| 缓存命中压测（32 并发 / 512 请求） | **0 错误**，77.0 req/s，P50 149ms / **P95 268ms** / P99 331ms |
| Redis 全挂 | 仍 **200**，`cache_source="local"`，248ms |
| LLM 全挂 | 仍 **200**，2326 字降级报告，`degraded=true` |
| 重试 A/B（260 次，40% 注入失败） | 成功率 **55.8% → 90.4%**（调用数 260 → 416） |
| 冷路径（真模型） | 中位 **20.6s**（`/analyze` ≈ 2~4s + `/confirm` ≈ 16~22s），报告 3.8k~4.4k 字 |

---

## 八、已知取舍（写在明面上）

- **异步用 Redis Stream 而非 Celery/RabbitMQ**：环境装不了 Celery，但 Redis 现成，于是手写 Redis Stream 真队列（XREADGROUP 消费组 / XAUTOCLAIM 超时接管 / 死信 / 幂等去重），`jobs.submit/get` 接口不变。代价是 at-least-once 而非 exactly-once，幂等要自己兜。
- **指标是进程内自建的，不是 Prometheus**：环境装不了 prometheus_client，`snapshot()` 换成 `Counter/Histogram` 时语义一致、结构不用动。
- **覆盖率工具是自己写的**：`coverage.py` 同样装不上，用 `compile()` 取可执行行 + `sys.settrace` 统计命中，误差来自"装饰器/多行语句"的行归属，量级可接受。
- **集成测试用假模型**：`mock.patch` 只在测试进程内生效，生产仍走真 DeepSeek；真模型链路由 `tools/e2e_live.py` 单独验证。

---

## 九、Roadmap

- [x] 接入真实公开数据（TCGA-BRCA，226 样本 × 约 2 万基因）
- [x] 持久化换成 PostgresSaver / PostgresStore（跨重启不丢）
- [x] 富集分析（GO / KEGG）
- [x] Human-in-the-loop：关键步骤人工确认
- [x] 长期记忆：记住用户偏好阈值
- [x] Streamlit 网页界面 + FastAPI 后端 + 账号体系 + Docker 部署
- [x] 工程化加固：缓存一致性 / 降级 / 熔断重试 / 上下文预算 / 异步任务 / 可观测
- [x] 自动化测试 + 覆盖率 + 压测与故障注入
- [x] Fork 分叉：同一份数据用不同阈值跑两条分支对比
- [x] 把 `jobs` 换成 Redis Stream 真队列（任务态跨进程、kill -9 不丢）
- [ ] 指标接 Prometheus + Grafana（当前 `snapshot()` 已按此设计）

---

## 十、技术栈

Python · LangGraph · LangChain · DeepSeek · bge-m3 · Chroma · FastAPI · SQLAlchemy · MySQL · PostgreSQL · Redis · JWT · bcrypt · pandas · NumPy · SciPy · Docker · Streamlit

---

## 十一、相关文档

- [工程化升级说明](docs/工程化升级说明.md)：这次加固的每个决策、踩坑与复现命令
- [简历项目条目](docs/简历项目条目.md)：可直接用的项目描述与面试口径
- [项目总览](docs/项目总览.md) / [项目详解](docs/项目详解.md) / [面试知识点](docs/面试知识点.md)

<!-- git 练习：2026-11-25 -->
