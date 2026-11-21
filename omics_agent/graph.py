"""把节点拼成一张图，加上条件路由、检查点、长期记忆。

图的结构：
  START → parse_intent
              ├─ analysis_type=other → report（暂不支持）
              ├─ analysis_type=pca → pca_analysis → report → END
              ├─ analysis_type=clustering → cluster_analysis → report → END
              └─ analysis_type=differential → load_qc
                        ├─ 质检不合格 → report（质检失败）
                        └─ 质检合格 → human_review（人工确认）
                                    ├─ 用户取消 → report
                                    └─ 用户批准 → diff_analysis → enrichment
                                                  → interpret → save_memory → report → END

持久化：默认内存（InMemorySaver + InMemoryStore），设 USE_POSTGRES=true 切 PostgreSQL。
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from omics_agent import config, nodes
from omics_agent.state import AnalysisState


def route_after_intent(state: dict) -> str:
    """① 之后的分叉：按分析类型路由到不同节点。"""
    analysis_type=state.get("analysis_type")
    if analysis_type == "differential":
        return "load_qc"
    if analysis_type == "pca":
        return "pca_analysis"
    if analysis_type == "clustering":
        return "cluster_analysis"
    if analysis_type == "qa":  # ← 新加：知识库问答
        return "qa_node"
    return "report"  # other 或未知 → 不支持


def route_after_qc(state: dict) -> str:
    """② 质检之后的分叉：合格先进人工确认，不合格直接出报告。"""
    return "human_review" if state.get("qc_passed", False) else "report"


def route_after_review(state: dict):
    """③ 人工确认之后：批准 → Send 分叉两条 diff_analysis（不同阈值）；取消 → 报告。

    Send 让 human_review 一次发出多个并行任务到 diff_analysis，各带一组
    fc/fdr 阈值，跑完由 compare_thresholds 汇合对比（LangGraph 自动并行 + join）。
    """
    if not state.get("qc_passed", False):
        return "report"  # 取消：单字符串，直接走报告

    fc = state.get("fc_threshold", 1.0)
    fdr = state.get("fdr_threshold", 0.05)
    return [
        # 主阈值：用户/记忆的当前阈值（后续富集、解读用它的结果）
        Send("diff_analysis", {**state, "fc_threshold": fc, "fdr_threshold": fdr}),
        # 严格阈值：fc 翻倍、fdr 收紧 5 倍，用来对比阈值松紧对结果的影响
        Send("diff_analysis", {**state, "fc_threshold": round(fc * 2, 2), "fdr_threshold": round(fdr / 5, 4)}),
    ]


def build_graph():
    builder = StateGraph(AnalysisState)

    # 加节点pca等
    builder.add_node("pca_analysis", nodes.pca_analysis_node)
    builder.add_node("cluster_analysis", nodes.cluster_analysis_node)

    #Rag
    builder.add_node("qa_node", nodes.qa_node)  # ← 新加

    builder.add_node("parse_intent", nodes.parse_intent_node)
    builder.add_node("load_qc", nodes.load_qc_node)
    builder.add_node("human_review", nodes.human_review_node)
    builder.add_node("diff_analysis", nodes.diff_analysis_node)
    builder.add_node("compare_thresholds", nodes.compare_thresholds_node)  # ← Fork 汇合点
    builder.add_node("enrichment", nodes.enrichment_node)
    builder.add_node("interpret", nodes.interpret_node)
    builder.add_node("save_memory", nodes.save_memory_node)
    builder.add_node("report", nodes.report_node)

    # 加边
    builder.add_edge(START, "parse_intent")

    # 条件边 1：按分析类型路由
    # add_conditional_edges 三参数：源节点 / 路由函数（返回字符串）/ 路由返回值→目标节点的映射
    builder.add_conditional_edges(
        "parse_intent", route_after_intent,
        {"load_qc": "load_qc",
         "pca_analysis": "pca_analysis",
         "cluster_analysis": "cluster_analysis",
         "qa_node": "qa_node",
         "report": "report",
         },
    )

    # 条件边 2：质检合格 → 人工确认，否则 → 报告
    builder.add_conditional_edges(
        "load_qc", route_after_qc,
        {"human_review": "human_review", "report": "report"},
    )

    # 条件边 3：人工确认批准 → 分析，取消 → 报告
    builder.add_conditional_edges(
        "human_review", route_after_review,
        {"diff_analysis": "diff_analysis", "report": "report"},
    )

    builder.add_edge("pca_analysis", "report")
    builder.add_edge("cluster_analysis", "report")

    builder.add_edge("qa_node", "report")  # ← 新加：问答完直接出报告

    builder.add_edge("diff_analysis", "compare_thresholds")  # 并行分支先汇合对比
    builder.add_edge("compare_thresholds", "enrichment")
    builder.add_edge("enrichment", "interpret")
    builder.add_edge("interpret", "save_memory")
    builder.add_edge("save_memory", "report")
    builder.add_edge("report", END)

    # 检查点 + 长期记忆：默认内存，设 USE_POSTGRES=true 切到 PostgreSQL
    return builder.compile(
        checkpointer=config.get_checkpointer(),
        store=config.get_store(),
    )
