"""多工具 Agent（第三步）：用 LangGraph 手动搭 ReAct 循环，让 LLM 自主调用工具。

和固定图（graph.py）的区别，一句话：
  固定图 = 你（代码）在入口问一次 LLM「这是哪种分析」，然后走写死的条件边；
  ReAct 循环 = 每轮都由 LLM 自己决定「调哪个工具 / 还是不调了直接回答」，
              直到它觉得信息够了才停。这就是「Agent」的自主性来源。

为什么手动搭而不是 create_react_agent：
  create_react_agent 一行就好，但底层被藏起来了，面试讲不清；手动搭只有
  5 个零件（状态/模型/agent 节点/tool 节点/条件边），每个都能说清楚原理。

结构：
  START → agent（LLM 决定） ⇄ tools（执行工具） → END
  - agent 节点：把「问题 + 历史消息」喂给绑了工具的模型，模型返回
    AIMessage——要么带 tool_calls（我想调工具），要么是最终答案；
  - tools 节点：真正执行工具，把结果作为 ToolMessage 塞回对话；
  - 条件边 should_continue：模型还想调工具 → 回 tools；否则 → END。

「跑飞」防护（面试必问「你的 Agent 死循环怎么办」，答案就在这几层）：
  1. max_steps：工具轮数预算，用尽后拿掉工具、强制模型「用已有信息作答」；
  2. 重复调用检测：(工具名, 参数) 指纹去重，原地打转的调用不再真的执行；
  3. 单工具超时：某个工具卡住只放弃这一次等待，不拖死整个请求；
  4. GraphRecursionError：框架级最后一道防线，兜成可读的降级回答。
  参数默认值集中在下面「ReAct 护栏参数」一节，可用环境变量覆盖。

长期记忆（Store）：
  编译图时挂上 store，ReAct 侧因此有了 recall_preferences / save_preference
  两个工具，和固定图（nodes.py 的 save_memory_node）共用同一个命名空间，
  所以「ReAct 里记住的阈值偏好」固定图那边读得到，反之亦然。

目前有 11 个工具（5 个分析/检索 + 2 个 Skill 技能库 + 2 个长期记忆 + 2 个 MCP 数据库）：
  query_gene            基因知识问答（先精确查名，再语义检索）
  run_differential      差异表达分析（找显著上/下调基因）
  run_pca               主成分分析（看样本分布）
  run_clustering        KMeans 聚类（看样本分几群）
  search_literature     检索 PubMed 文献（联网，拿标题+摘要）
  list_skills           列出可用的技能（Skill）清单
  load_skill            加载某个技能的完整说明（SKILL.md 正文）
  recall_preferences    读某位研究者长期记忆里的分析偏好
  save_preference       把一条分析偏好写进长期记忆
  mcp_get_user_count    统计注册用户数（MCP 协议连数据库 server）
  mcp_get_history       查某用户最近分析历史（MCP 协议连数据库 server）

用法（项目根目录）：
    python -B omics_agent/agent.py                       # 默认跑一个组合问题
    python -B omics_agent/agent.py "做个 PCA"           # 传自己的问题
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from concurrent.futures import TimeoutError as FutureTimeout  # noqa: E402

import numpy as np  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402

from omics_agent.config import get_checkpointer, get_model, get_store  # noqa: E402
from omics_agent.dataio import read_csv_cached  # noqa: E402
from omics_agent.diff import differential_expression  # noqa: E402
from omics_agent.mcp_client import MCP_TOOLS  # noqa: E402
# MEMORY_NS / PREFS_KEY 直接从固定图那边引进来：长期记忆只有一个命名空间，
# 两边各写一份常量迟早会漂移，读到的偏好就对不上了。
from omics_agent.nodes import (  # noqa: E402
    DATASET_PATH,
    MEMORY_NS,
    PREFS_KEY,
    discover_dataset,
)
from omics_agent.rag import get_gene_function, retrieve_genes  # noqa: E402
from omics_agent.resilience import call_with_retry  # noqa: E402

# 抽「疑似基因名」的正则（和 nodes.py 里的一样）：BRCA1、TP53 这类
_GENE_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,14}")


# ── 工具 1：基因知识问答 ──
@tool
def query_gene(question: str) -> str:
    """查询基因知识库，回答基因功能类问题。

    当用户问「某个基因是什么 / 有什么功能 / 哪些基因参与某条通路」这类
    生物学知识问题时调用。question 是用户的问题原文。
    """
    # ① 先精确查名：问题里若含明确基因名（如 BRCA1），直接拿它的功能原文，
    #    比语义检索准（语义检索会返回「相似的」基因，可能张冠李戴）。
    for token in _GENE_RE.findall(question or ""):
        try:
            desc = get_gene_function(token.upper())
        except Exception:  # noqa: BLE001 - 知识库读失败不致命，退到语义检索
            desc = None
        if desc:
            return f"{token.upper()}：{desc}"
    # ② 查不到精确名，再语义检索（适合「哪些基因参与 DNA 修复」这类问题）
    #    embedding 也是网络调用，会抖，套一次重试
    hits = call_with_retry(lambda: retrieve_genes(question, top_k=5), attempts=2, label="retrieve")
    if not hits:
        return "知识库里没检索到相关基因，换个问法试试。"
    return "\n".join(f"- {g}：{doc[:300]}（相似度 {s:.2f}）" for g, doc, s in hits)


# ── 工具 2：差异表达分析 ──
@tool
def run_differential() -> str:
    """对表达矩阵做差异表达分析，找出两组样本间显著上/下调的基因。

    当用户想找差异基因、显著基因、上调/下调基因、火山图时调用。
    """
    try:
        found = discover_dataset(DATASET_PATH)
        expr_df = read_csv_cached(found["expr_file"])
        meta_df = read_csv_cached(found["meta_file"])
        group_col = found["group_col"]
        groups = meta_df[group_col].unique().tolist()
        res = differential_expression(
            expr_df, meta_df, group_col, group_a=groups[0], group_b=groups[1]
        )
    except Exception as e:  # noqa: BLE001 - 数据缺失/列名不符，返回友好报错交还模型
        return f"差异表达分析失败：{e}"
    sig = res[res["significant"]]
    if sig.empty:
        return "未找到显著差异表达基因。"
    up = int((sig["log2fc"] > 0).sum())
    down = int((sig["log2fc"] < 0).sum())
    top = sig.head(10)[["gene", "log2fc", "fdr"]].to_string(index=False)
    return f"显著差异基因共 {len(sig)} 个（上调 {up}、下调 {down}）。Top10：\n{top}"


# ── 工具 3：主成分分析 ──
@tool
def run_pca() -> str:
    """对表达矩阵做主成分分析(PCA)，看样本在低维空间的分布。

    当用户想看样本分布、降维、样本间关系时调用。
    """
    try:
        found = discover_dataset(DATASET_PATH)
        expr_df = read_csv_cached(found["expr_file"])
        X = expr_df.iloc[:, 1:].values.astype(float).T  # 转置成 样本×基因
        X = X - X.mean(axis=0)  # 中心化
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except Exception as e:  # noqa: BLE001
        return f"PCA 分析失败：{e}"
    explained = (S ** 2) / (S ** 2).sum()
    return (
        f"PCA 完成：{X.shape[0]} 个样本，PC1 解释 {explained[0]:.1%}，"
        f"PC2 解释 {explained[1]:.1%}，PC3 解释 {explained[2]:.1%}"
        f"（前三合计 {explained[:3].sum():.1%}）"
    )


# ── 工具 4：KMeans 聚类 ──
@tool
def run_clustering(k: int = 3) -> str:
    """对样本做 KMeans 聚类，看能分成几群。

    当用户想看样本分组、分群、亚型时调用。k 是要分的群数，默认 3。
    """
    try:
        found = discover_dataset(DATASET_PATH)
        expr_df = read_csv_cached(found["expr_file"])
        X = expr_df.iloc[:, 1:].values.astype(float).T
        np.random.seed(42)  # 固定随机种子，结果可复现
        centers = X[np.random.choice(len(X), k, replace=False)]
        for _ in range(100):
            dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
    except Exception as e:  # noqa: BLE001
        return f"聚类失败：{e}"
    counts = np.bincount(labels, minlength=k)
    return "聚类完成（k=%d）：" % k + "、".join(f"簇 {i} = {counts[i]} 个样本" for i in range(k))


# ── 工具 5：检索 PubMed 文献 ──
@tool
def search_literature(query: str, max_results: int = 3) -> str:
    """检索 PubMed 生物医学文献，返回相关论文的标题和摘要。

    当用户想查某个主题/基因/疾病的最新文献、研究进展、相关论文时调用。
    query 是检索词（英文效果最好），例如 "BRCA1 breast cancer"。
    """
    def _fetch() -> str:
        # 把「取数据」的整段逻辑包进内部函数，好让重试逻辑只盯着它
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        # ① esearch：关键词 → PMID（论文编号）列表
        url = f"{base}/esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmax={max_results}&retmode=json"
        with urllib.request.urlopen(url, timeout=15) as r:
            ids = json.load(r).get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "没有检索到相关文献，换个检索词试试。"

        # ② efetch：用 PMID 批量拿标题 + 摘要（XML）
        url = f"{base}/efetch.fcgi?db=pubmed&id={','.join(ids)}&rettype=abstract&retmode=xml"
        with urllib.request.urlopen(url, timeout=15) as r:
            root = ET.fromstring(r.read().decode("utf-8"))

        # ③ 解析每篇论文：标题 / 期刊 / 年份 / 摘要
        out = []
        for art in root.findall(".//PubmedArticle"):
            title_el = art.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            journal_el = art.find(".//Journal/Title")
            journal = "".join(journal_el.itertext()).strip() if journal_el is not None else ""
            year_el = art.find(".//PubDate/Year")
            year = "".join(year_el.itertext()).strip() if year_el is not None else ""
            abstract = " ".join(
                (t.text or "").strip() for t in art.findall(".//Abstract/AbstractText")
            )
            out.append(f"《{title}》（{journal}, {year}）\n摘要：{abstract[:400]}")
        return "\n\n".join(out)

    # 网络抖动（超时/502）是「临时失败」：让工具自己重试，都失败才抛给模型
    return call_with_retry(_fetch, attempts=2, label="literature")


# ── Skill 技能库：把「固定工作流」打包成 SKILL.md，模型用到才加载 ──
SKILLS_DIR = os.path.join(ROOT, "skills")


def _read_skill_description(skill_md: str) -> str:
    """从 SKILL.md 里抽一行 description 做列表摘要，读不到就留空。"""
    try:
        with open(skill_md, encoding="utf-8") as f:
            for line in f:
                if line.startswith("description:"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001 - 技能文件读不到不致命，返回空描述
        return ""
    return ""


# ── 工具 6：列出技能清单 ──
@tool
def list_skills() -> str:
    """列出当前可用的技能（Skill）清单，每项一行「技能名：一句话说明」。

    当用户想先看看「你能做什么 / 有哪些工作流」时调用；
    或不确定该用哪个技能时，先列出来让用户选。
    """
    if not os.path.isdir(SKILLS_DIR):
        return "技能库为空（skills 目录不存在）。"
    lines = []
    for name in sorted(os.listdir(SKILLS_DIR)):
        skill_md = os.path.join(SKILLS_DIR, name, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue
        desc = _read_skill_description(skill_md)
        lines.append(f"- {name}：{desc}" if desc else f"- {name}")
    return "\n".join(lines) if lines else "技能库为空。"


# ── 工具 7：加载某个技能 ──
@tool
def load_skill(name: str) -> str:
    """加载某个技能的完整说明（SKILL.md 正文），返回该技能的操作步骤。

    当确定要执行某个技能时调用。name 是技能名（如 differential），
    先调 list_skills 看有哪些技能、名字分别是什么。
    """
    skill_md = os.path.join(SKILLS_DIR, name, "SKILL.md")
    if not os.path.isfile(skill_md):
        return f"没有名为「{name}」的技能，先调 list_skills 看看有哪些。"
    with open(skill_md, encoding="utf-8") as f:
        return f.read()


# ══════════ 长期记忆（Store）：让 ReAct 侧也能读写「研究者偏好」══════════
# 命名空间/键直接用 nodes.py 里的 MEMORY_NS / PREFS_KEY：
# 固定图存的是 (MEMORY_NS + user_id, PREFS_KEY) → {fc_threshold, fdr_threshold, ...}，
# ReAct 这里读写的完全是同一个槽位，两条入口共享一份长期记忆。

# 合法字段 → 类型。写回时按这个表做类型纠正，保证 ReAct 写进去的形状
# 和固定图 save_memory_node 写进去的形状一致（否则固定图读到 str 阈值会炸）。
PREF_FIELDS = {
    "fc_threshold": float,
    "fdr_threshold": float,
    "group_a": str,
    "group_b": str,
}

# build_agent_graph 编译时把 store 实例记在这里，供「不在图里执行」的场景兜底
_STORE = None


def _memory_store():
    """取长期记忆 Store（优先拿运行期注入的那个实例）。

    为什么要有这个函数：
      图里的节点拿 store 很方便（runtime.store），但工具函数是普通函数，
      没有 runtime 参数。LangGraph 的约定是：compile(store=...) 之后，
      节点内执行时可以用 langgraph.config.get_store() 拿到同一个实例。
      老版本没有这个 API、或者单独 import 工具做单测时拿不到，就退回
      模块里缓存的那个实例——两条路都指向「同一份记忆」，不会出现
      「写进去了但读不到」的假象。
    """
    global _STORE
    try:
        from langgraph.config import get_store as _lg_get_store

        injected = _lg_get_store()
        if injected is not None:
            return injected
    except Exception:  # noqa: BLE001 - API 不存在/不在图上下文里，退回单例
        pass
    if _STORE is None:
        _STORE = get_store()   # config.get_store()：内存版或 PostgresStore
    return _STORE


def _coerce_pref(key: str, value):
    """把用户/模型给的值转成该字段原本的类型（阈值存成 str 会让下游比较报错）。"""
    want = PREF_FIELDS[key]
    if want is float:
        return float(value)
    return str(value)


# ── 工具 8：读长期记忆里的偏好 ──
@tool
def recall_preferences(user_id: str = "default") -> str:
    """读这位研究者长期记忆里的分析偏好（阈值、上次用的分组）。

    当用户说「按我上次的口径来 / 我习惯的阈值是多少」，或者你想知道
    之前给这个人用什么阈值做的差异分析时调用。user_id 是用户标识。
    """
    try:
        store = _memory_store()
        item = store.get((*MEMORY_NS, user_id), PREFS_KEY)
    except Exception as e:  # noqa: BLE001 - 记忆读失败不该让整轮对话崩掉
        return f"读取长期记忆失败：{e}"
    if item is None:
        return f"长期记忆里还没有「{user_id}」的偏好记录（第一次来就是空的，正常）。"
    # store 返回的是 Item(value=...)，也可能直接是 dict（不同后端略有差异）
    value = getattr(item, "value", item)
    if not isinstance(value, dict) or not value:
        return f"长期记忆里「{user_id}」偏好为空或格式不认识：{value!r}"
    body = "、".join(f"{k}={v}" for k, v in value.items() if k in PREF_FIELDS)
    return f"长期记忆里「{user_id}」的偏好：{body or value}"


# ── 工具 9：写一条偏好进长期记忆 ──
@tool
def save_preference(user_id: str, key: str, value: str) -> str:
    """把一条分析偏好写进长期记忆，下次分析自动复用。

    当用户明确表达了偏好（如「以后都用 FDR 0.01」「我一般叫 case/control」）
    时调用。key 只能是 fc_threshold / fdr_threshold / group_a / group_b；
    value 传字符串，这里会按字段类型转换后写回。
    """
    if key not in PREF_FIELDS:
        return f"不认识的偏好字段「{key}」，只支持：{', '.join(PREF_FIELDS)}"
    try:
        store = _memory_store()
    except Exception as e:  # noqa: BLE001
        return f"长期记忆不可用，本次偏好没有保存：{e}"
    namespace = (*MEMORY_NS, user_id)
    try:
        # 先读再合并：store.put 是整条覆盖，不合并的话会把别的偏好字段抹掉
        item = store.get(namespace, PREFS_KEY)
        current = dict(getattr(item, "value", item) or {}) if item is not None else {}
        current[key] = _coerce_pref(key, value)
        store.put(namespace, PREFS_KEY, current)
    except Exception as e:  # noqa: BLE001 - 记忆写失败不影响主流程
        return f"写长期记忆失败：{e}"
    return f"已记住：{user_id} 的 {key} = {current[key]}（当前全部偏好：{current}）"


TOOLS = [query_gene, run_differential, run_pca, run_clustering, search_literature,
         list_skills, load_skill, recall_preferences, save_preference, *MCP_TOOLS]


# ══════════ ReAct 护栏参数（可用环境变量覆盖）══════════
# 为什么要有这几行：LangGraph 默认的 recursion_limit=25 是「super-step 上限」，
# 一个工具轮 = agent + tools 两个 super-step，所以 25 大约等于 12 轮工具调用；
# 而且一旦超限它抛 GraphRecursionError——如果不 catch，用户看到的是 traceback
# 而不是回答。所以这里自己定义预算，并让框架上限与它对齐。
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))           # 最多几轮工具调用
TOOL_TIMEOUT = float(os.getenv("AGENT_TOOL_TIMEOUT", "30"))  # 单工具最长执行时间（秒）
MAX_REPEAT = int(os.getenv("AGENT_MAX_REPEAT", "1"))         # 同一 (工具, 参数) 最多执行几次

# 预算用尽时喂给模型的指令。注意：此时模型是「不带工具」的实例，
# 从协议层面就拿掉了 tool_calls 能力，比只在提示词里求它别调工具可靠得多。
LIMIT_NOTE = (
    "【护栏】本轮最多允许 {max_steps} 次工具调用，现在已经用完。"
    "请不要、也无法再调用任何工具，只能基于上面已经拿到的工具结果直接回答用户；"
    "对于没有拿到的信息，明确写「未获取到」，并说明还需要哪一步（例如再做一次差异分析）才能补齐。"
)


class AgentState(MessagesState):
    """ReAct 状态：在 messages 之外多记两样东西给护栏用。

    MessagesState 已经定义了 messages（带 add_messages reducer，新消息是追加）；
    下面三个字段走 LastValue 通道（后写的覆盖先写的），所以节点里要返回
    「合并后的完整值」，不能只返回增量。
    """
    steps: int              # 已用掉的步数预算（= 已尝试过的工具调用次数）
    calls: dict[str, int]   # 调用指纹 → 已执行次数，用来发现「原地打转」
    guard_note: str         # 护栏触发时的说明，最终结果里带给调用方


def _call_fingerprint(tool_name: str, args) -> str:
    """工具调用指纹：同名工具 + 同样的参数 = 同一个指纹。

    用 json.dumps(sort_keys=True) 而不是 hash(args)：
      · dict 本身不可哈希；
      · sort_keys 保证 {"a":1,"b":2} 与 {"b":2,"a":1} 指纹相同——模型重排参数
        顺序是很常见的，按顺序不同判成不同调用就去不了重了。
    """
    try:
        payload = json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - 参数里有不可序列化对象时退回 repr
        payload = repr(args)
    return f"{tool_name}|{payload}"


# 工具执行池：给单工具超时用的。注意 Python 线程无法被强杀，超时只是
# 「不再等它」——worker 会把这次调用跑完再释放。所以 max_workers 给到 4，
# 即使偶尔卡死一个也不会让后续调用排队排到天荒地老。
_TOOL_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="react-tool")

# 工具名 → 工具对象，避免每次调用都线性扫一遍列表
_TOOL_BY_NAME = {t.name: t for t in TOOLS}


def _run_one_tool(call: dict) -> str:
    """执行单个工具调用；「工具不存在 / 参数错 / 超时 / 抛异常」全变成文本。

    为什么全都返回字符串而不是抛异常：ToolMessage 里塞一段可读的失败原因，
    模型下一轮就能自己纠（换个参数、换个工具、或者干脆基于已有信息作答），
    这比把异常抛到图外面、让用户看 500 要好得多。
    """
    name = call.get("name") or ""
    args = call.get("args") or {}
    fn = _TOOL_BY_NAME.get(name)
    if fn is None:
        return (f"没有名为「{name}」的工具。可用工具："
                f"{', '.join(sorted(_TOOL_BY_NAME))}")
    if not isinstance(args, dict):
        return f"工具 {name} 的参数必须是对象，收到的是：{args!r}"
    try:
        future = _TOOL_POOL.submit(lambda: str(fn.invoke(args)))
        return future.result(timeout=TOOL_TIMEOUT)
    except FutureTimeout:
        return (f"工具 {name} 超过 {TOOL_TIMEOUT:.0f} 秒未返回，本次调用已放弃；"
                "可以换更小的参数或换一个工具再试。")
    except Exception as e:  # noqa: BLE001 - 参数校验失败、内部异常都交还模型
        return f"工具 {name} 执行失败：{e}"


def tools_node(state: AgentState) -> dict:
    """执行模型这一轮点名的工具，把结果作为 ToolMessage 交还。

    为什么不用现成的 ToolNode（这也是必须自己写的原因）：
      ① 去重：同一个 (工具, 参数) 反复调用只会收到「已执行过」的提示，
         直接掐掉「模型原地打转」这种最常见的浪费；
      ② 单工具超时：某个工具（比如联网检索）卡住时不至于拖死整个请求；
      ③ 计数：把用掉的步数写回 state，供 agent 节点判断预算。
    ToolNode 能做的「异常变 ToolMessage」这里同样做了（见 _run_one_tool）。
    """
    last = state["messages"][-1]
    calls = list(getattr(last, "tool_calls", None) or [])
    seen = dict(state.get("calls") or {})
    steps = int(state.get("steps") or 0)
    out: list[ToolMessage] = []

    for call in calls:
        name = call.get("name") or ""
        fingerprint = _call_fingerprint(name, call.get("args") or {})
        used = seen.get(fingerprint, 0)
        if used >= MAX_REPEAT:
            # 拦掉但不扣「真实执行」的冤枉时间；提示里给出已有次数，让模型自己转方向
            content = (f"重复调用已被拦截：{name} 用完全相同的参数已经执行过 {used} 次，"
                       "结果就在上面的工具消息里。请直接基于已有信息作答，或换参数/换工具。")
        else:
            seen[fingerprint] = used + 1
            content = _run_one_tool(call)
        # 每个 tool_call 都必须回一条 ToolMessage（哪怕是被拦掉的），
        # 否则下一轮把带 tool_calls 的 AIMessage 发给 API 会因为缺响应报 400。
        out.append(ToolMessage(content=content, tool_call_id=call.get("id") or fingerprint,
                               name=name))
        # 被拦截的调用也计入步数：否则「模型一直重复同一个调用」时步数不涨，
        # 预算永远用不完，护栏就形同虚设。
        steps += 1

    return {"messages": out, "steps": steps, "calls": seen}


def build_agent_graph():
    """搭 ReAct 循环图，返回编译好的图。"""
    global _STORE
    # bind_tools = 把工具清单「告诉」模型；此后模型才能输出 tool_calls
    model = get_model().bind_tools(TOOLS)
    # 不绑工具的同一个模型：步数用尽时用它，从协议层面禁用 tool_calls
    plain_model = get_model()

    def agent_node(state: AgentState) -> dict:
        """把整段对话喂给模型，返回模型的新回复（追加进 messages）。

        这里同时是「预算闸门」：步数用尽后换 plain_model，强制模型用已有
        信息作答——这就是超限时的降级路径（而不是抛异常或死循环）。
        """
        steps = int(state.get("steps") or 0)
        messages = list(state["messages"])
        if steps >= MAX_STEPS:
            note = LIMIT_NOTE.format(max_steps=MAX_STEPS)
            return {
                "messages": [plain_model.invoke(messages + [HumanMessage(content=note)])],
                "guard_note": note,
            }
        return {"messages": [model.invoke(messages)]}

    def should_continue(state: AgentState) -> str:
        """路由：模型最后一条消息带 tool_calls → 去执行工具；否则结束。

        注意这里不检查步数预算：预算的闸门放在 agent_node 里（进来前就判），
        因为如果在这里把「还带着 tool_calls 的消息」直接送 END，那条 AIMessage
        就永远等不到 ToolMessage，历史不完整，下次 invoke 会被 API 判 400。
        """
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")  # 工具跑完，交还给 agent 再决策（循环）
    # 挂检查点：同一 thread_id 多次 invoke 会接着上次对话继续，中断也能续跑
    # 挂 store：长期记忆（ReAct 侧的 recall_preferences / save_preference 靠它）
    _STORE = get_store()
    return builder.compile(checkpointer=get_checkpointer(), store=_STORE)


def default_recursion_limit() -> int:
    """把 LangGraph 的 super-step 上限和我们自己的步数预算对齐。

    一轮 ReAct = agent + tools 两个 super-step，再留 4 步余量给收尾/降级。
    显式设置的好处：不会出现「我们的护栏还没触发，LangGraph 先抛
    GraphRecursionError」这种两套阈值打架的局面。
    """
    return MAX_STEPS * 2 + 4


def run_react(question: str, thread_id: str = "agent-demo",
              recursion_limit: int | None = None) -> dict:
    """跑一次 ReAct，并把「图跑飞」这种框架级异常收成可读结果。

    返回的 dict 里一定有 messages；如果撞上了框架上限，还会有
    guard_note="react_recursion_limit"，调用方（web.py 的 /agent）据此
    决定是提示用户拆问题还是照常展示。
    """
    graph = build_agent_graph()
    limit = recursion_limit or default_recursion_limit()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": limit,
    }
    initial = {"messages": [HumanMessage(question)], "steps": 0, "calls": {}}
    try:
        return graph.invoke(initial, config=config)
    except GraphRecursionError as exc:
        # 最后一道防线：即使护栏本身出问题（比如历史被外部塞得很长），
        # 也不能把 traceback 甩给用户，要给一条能读懂的降级说明。
        return {
            "messages": [AIMessage(content=(
                f"这个问题需要的工具轮数超过了安全上限（recursion_limit={limit}），"
                "已中止本次自动探索。基于目前拿到的信息无法给出可靠结论，"
                f"建议把问题拆小后重问。（内部错误：{exc}）"))],
            "steps": 0,
            "calls": {},
            "guard_note": "react_recursion_limit",
        }


def main() -> int:
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "先做主成分分析看看样本分布，再找出两组样本之间显著差异表达的基因"
    )
    print(f"问：「{question}」\n")
    # thread_id 是「会话钥匙」：同一 id 再次 invoke，会从上次的检查点接着聊
    result = run_react(question, thread_id="agent-demo")

    # 逐条打印消息，看清 ReAct 的「想 → 调工具 → 看结果 → 再想 → 回答」过程
    for i, msg in enumerate(result.get("messages", []), 1):
        name = type(msg).__name__
        if getattr(msg, "tool_calls", None):
            for c in msg.tool_calls:
                print(f"[{i}] {name} → 要调工具：{c['name']}({c['args']})")
        elif name == "ToolMessage":
            print(f"[{i}] ToolMessage → 工具返回：{str(msg.content)[:120]}...")
        else:
            print(f"[{i}] {name} → {str(msg.content)}")

    # 护栏触发时要让操作者看见，别把降级回答当成正常结论
    steps = result.get("steps", 0)
    print(f"\n本次共尝试 {steps} 次工具调用（预算 {MAX_STEPS} 次）。")
    if result.get("guard_note"):
        print(f"⚠️ 护栏触发：{result['guard_note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
