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

目前有 5 个工具（4 种分析能力 + 1 个联网文献检索）：
  query_gene          基因知识问答（先精确查名，再语义检索）
  run_differential    差异表达分析（找显著上/下调基因）
  run_pca             主成分分析（看样本分布）
  run_clustering      KMeans 聚类（看样本分几群）
  search_literature   检索 PubMed 文献（联网，拿标题+摘要）

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

import numpy as np  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from omics_agent.config import get_checkpointer, get_model  # noqa: E402
from omics_agent.dataio import read_csv_cached  # noqa: E402
from omics_agent.diff import differential_expression  # noqa: E402
from omics_agent.nodes import DATASET_PATH, discover_dataset  # noqa: E402
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


TOOLS = [query_gene, run_differential, run_pca, run_clustering, search_literature]


def build_agent_graph():
    """搭 ReAct 循环图，返回编译好的图。"""
    # bind_tools = 把工具清单「告诉」模型；此后模型才能输出 tool_calls
    model = get_model().bind_tools(TOOLS)

    def agent_node(state):
        """把整段对话喂给模型，返回模型的新回复（追加进 messages）。"""
        return {"messages": [model.invoke(state["messages"])]}

    def should_continue(state):
        """路由：模型最后一条消息带 tool_calls → 去执行工具；否则结束。"""
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    # handle_tool_errors=True：工具抛异常时，ToolNode 把异常变成 ToolMessage 交还模型，
    # 而不是直接崩溃。这样「工具真失败」也走「模型再决策」的优雅路径。
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")  # 工具跑完，交还给 agent 再决策（循环）
    # 挂检查点：同一 thread_id 多次 invoke 会接着上次对话继续，中断也能续跑
    return builder.compile(checkpointer=get_checkpointer())


def main() -> int:
    graph = build_agent_graph()
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "先做主成分分析看看样本分布，再找出两组样本之间显著差异表达的基因"
    )
    print(f"问：「{question}」\n")
    # thread_id 是「会话钥匙」：同一 id 再次 invoke，会从上次的检查点接着聊
    config = {"configurable": {"thread_id": "agent-demo"}}
    result = graph.invoke({"messages": [HumanMessage(question)]}, config=config)

    # 逐条打印消息，看清 ReAct 的「想 → 调工具 → 看结果 → 再想 → 回答」过程
    for i, msg in enumerate(result["messages"], 1):
        name = type(msg).__name__
        if getattr(msg, "tool_calls", None):
            for c in msg.tool_calls:
                print(f"[{i}] {name} → 要调工具：{c['name']}({c['args']})")
        elif name == "ToolMessage":
            print(f"[{i}] ToolMessage → 工具返回：{str(msg.content)[:120]}...")
        else:
            print(f"[{i}] {name} → {str(msg.content)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
